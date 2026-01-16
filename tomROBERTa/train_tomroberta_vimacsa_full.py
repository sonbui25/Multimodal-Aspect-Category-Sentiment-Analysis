import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# train_tomroberta_vimacsa_full.py
# Full implementation of Multimodal TomBERT for ViMACSA
# Based on Official TomBERT Architecture (1-Layer Cross, 1-Layer Fusion)
# train_tomroberta_vimacsa_full.py
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler
# Remove deprecated autocast import if we use torch.amp.autocast directly
# from torch.cuda.amp import autocast 
from torch.autograd import Variable

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet152, ResNet152_Weights

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from text_preprocess import TextNormalize, convert_unicode

# ==============================================================================
# 1. SETUP LOGGING
# ==============================================================================
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

POLARITY_MAP = {0: 'None', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0.0
    )
    return p_macro, r_macro, f_macro

def save_model(path, model, optimizer, scheduler, epoch, best_score=0.0, scaler=None):
    if hasattr(model, 'module'): 
        model_state = model.module.state_dict()
    else: 
        model_state = model.state_dict()
        
    checkpoint_dict = {
        "epoch": epoch, 
        "best_score": best_score, 
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(), 
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler is not None: 
        checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint_dict, path)

# ==============================================================================
# 2. DATASET
# ==============================================================================

class TomBERTDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, aspects, num_img=3, num_roi=7):
        self.data = data 
        self.ASPECT = aspects
        self.pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224), antialias=True),  
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.roi_df = roi_df
        self.img_folder = img_folder
        self.num_img = num_img
        self.num_roi = num_roi
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['comment'])
        list_img_path = row['list_img']
        
        # --- Image ---
        list_img_features = []
        global_roi_features = []

        for img_path in list_img_path[:self.num_img]:
            full_path = os.path.join(self.img_folder, img_path)
            try:
                if os.path.exists(full_path):
                    one_image = read_image(full_path, mode=ImageReadMode.RGB)
                    img_tensor = self.transform(one_image).unsqueeze(0)
                else:
                    img_tensor = torch.zeros(1, 3, 224, 224)
                    one_image = torch.zeros(3, 224, 224)
            except:
                img_tensor = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224)
            list_img_features.append(img_tensor)
            
            # ROI
            current_rois = self.roi_df[self.roi_df['file_name'] == img_path].head(self.num_roi)
            list_roi_img = []
            if not current_rois.empty and one_image.shape[0] == 3:
                for _, roi_row in current_rois.iterrows():
                    try:
                        x1, y1, x2, y2 = int(roi_row['x1']), int(roi_row['y1']), int(roi_row['x2']), int(roi_row['y2'])
                        roi_crop = one_image[:, y1:y2, x1:x2]
                        if roi_crop.numel() > 0:
                            roi_tensor = self.transform(roi_crop).numpy()
                            list_roi_img.append(roi_tensor)
                    except: pass
            
            while len(list_roi_img) < self.num_roi:
                list_roi_img.append(np.zeros((3, 224, 224)))
            global_roi_features.append(list_roi_img)

        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        for i in range(min(len(list_img_features), self.num_img)):
            t_img_features[i] = list_img_features[i]

        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        for i in range(min(len(global_roi_features), self.num_img)):
            roi_img_features[i] = np.array(global_roi_features[i])

        # --- Text ---
        text_img_label = row.get('text_img_label', []) + row.get('text_label', [])
        labels_map = {asp: 0 for asp in self.ASPECT}
        for item in text_img_label:
            if '#' in item:
                parts = item.split('#')
                if len(parts) == 2:
                    asp, pol = parts
                    if "_" in asp: asp = "Public_area"
                    if asp in labels_map: labels_map[asp] = self.pola_to_num.get(pol, 0)

        out_tgt_ids, out_tgt_mask, out_sent_ids, out_sent_mask, out_labels = [], [], [], [], []

        for aspect in self.ASPECT:
            target_text = aspect.replace('_', ' ').lower()
            enc_tgt = self.tokenizer(target_text, max_length=16, padding='max_length', truncation=True)
            
            clean_review = text.replace('_', ' ').lower()
            enc_sent = self.tokenizer(text=clean_review, text_pair=target_text, max_length=170, padding='max_length', truncation=True)

            out_tgt_ids.append(torch.tensor(enc_tgt['input_ids']))
            out_tgt_mask.append(torch.tensor(enc_tgt['attention_mask']))
            out_sent_ids.append(torch.tensor(enc_sent['input_ids']))
            out_sent_mask.append(torch.tensor(enc_sent['attention_mask']))
            out_labels.append(labels_map[aspect])

        return (
            t_img_features, 
            torch.tensor(roi_img_features, dtype=torch.float32),
            torch.stack(out_tgt_ids), torch.stack(out_tgt_mask),
            torch.stack(out_sent_ids), torch.stack(out_sent_mask),
            torch.tensor(out_labels, dtype=torch.long), 
            text
        )

# ==============================================================================
# 3. MODELS
# ==============================================================================

class BERTLikePooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))

class TargetImageMatching(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size); self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
    def forward(self, target_feats, image_feats):
        attn_out, _ = self.mha(query=target_feats, key=image_feats, value=image_feats)
        h_v = self.norm1(target_feats + self.dropout(attn_out))
        h_v = self.norm2(h_v + self.feed_forward(h_v))
        return h_v

class TomBERT(nn.Module):
    def __init__(self, pretrained_path, num_labels=4):
        super(TomBERT, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_path)
        self.s2_roberta = AutoModel.from_pretrained(pretrained_path) 
        config = self.roberta.config
        self.hidden_size = config.hidden_size
        
        self.vis_projection = nn.Linear(2048, self.hidden_size)
        self.roi_projection = nn.Linear(2048, self.hidden_size)
        
        self.ti_matching = nn.ModuleList([TargetImageMatching(self.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob) for _ in range(1)])
        self.ent2img_pooler = BERTLikePooler(self.hidden_size)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob, activation="gelu", batch_first=True)
        self.mm_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.apply(self._init_custom_weights)

    def _init_custom_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding): module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): module.bias.data.zero_(); module.weight.data.fill_(1.0)

    def forward(self, target_ids, target_mask, sentence_ids, sentence_mask, visual_embeds_att, roi_embeds_att):
        s_out = self.roberta(sentence_ids, attention_mask=sentence_mask)
        h_s = s_out.last_hidden_state
        t_out = self.s2_roberta(target_ids, attention_mask=target_mask)
        h_t = t_out.last_hidden_state
        
        B, N_Img, Patches, Dim = visual_embeds_att.shape; _, _, N_Roi, _ = roi_embeds_att.shape
        vis_flat = visual_embeds_att.view(B, N_Img * Patches, Dim); roi_flat = roi_embeds_att.view(B, N_Img * N_Roi, Dim)
        g_visual = torch.cat([self.vis_projection(vis_flat), self.roi_projection(roi_flat)], dim=1)
        
        h_v = h_t
        for layer in self.ti_matching: h_v = layer(target_feats=h_v, image_feats=g_visual)
        h_v_pooled = self.ent2img_pooler(h_v).unsqueeze(1)
        
        mm_input = torch.cat([h_v_pooled, h_s], dim=1)
        bsz = sentence_mask.size(0)
        valid_cls = torch.ones(bsz, 1).to(sentence_mask.device)
        mm_mask = torch.cat([valid_cls, sentence_mask], dim=1)
        
        h_mm = self.mm_encoder(mm_input, src_key_padding_mask=(mm_mask == 0))
        logits = self.classifier(self.dropout(h_mm[:, 0, :]))
        return logits

class myResNetImg(nn.Module):
    def __init__(self, resnet, if_fine_tune):
        super().__init__()
        self.resnet = resnet; self.if_fine_tune = if_fine_tune
    def forward(self, x):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        att = F.adaptive_avg_pool2d(x, [7, 7])
        if not self.if_fine_tune: att = Variable(att.data)
        return att

class myResNetRoI(nn.Module):
    def __init__(self, resnet, if_fine_tune):
        super().__init__()
        self.resnet = resnet; self.if_fine_tune = if_fine_tune
    def forward(self, x):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2)
        if not self.if_fine_tune: fc = Variable(fc.data)
        return fc

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--image_dir', default='../vimacsa/image')
    parser.add_argument("--pretrained_hf_model", default="xlm-roberta-base", type=str)
    
    # Args
    parser.add_argument("--list_aspect", default=['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area'], nargs='+')
    parser.add_argument("--num_polarity", default=4, type=int)
    parser.add_argument("--num_imgs", default=3, type=int)
    parser.add_argument("--num_rois", default=7, type=int)
    
    # Training Params
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fine_tune_cnn', action='store_true')
    
    # Actions
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    
    args = parser.parse_args()
    
    # --- [FIX LOGIC] Update batch size BEFORE creating DataLoader ---
    if args.gradient_accumulation_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps: {args.gradient_accumulation_steps}, should be >= 1")
    
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    
    # Setup Device & Logging
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Running on device:{device}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    normalize_class = TextNormalize()
    
    roi_path = os.path.join(args.data_dir, "roi_data.csv")
    if os.path.exists(roi_path):
        roi_df = pd.read_csv(roi_path)
    else:
        roi_df = pd.DataFrame(columns=['file_name', 'x1', 'y1', 'x2', 'y2'])

    # Build Model
    model = TomBERT(args.pretrained_hf_model, num_labels=args.num_polarity).to(device)
    
    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(img_res_model, args.fine_tune_cnn).to(device)
    resnet_roi = myResNetRoI(roi_res_model, args.fine_tune_cnn).to(device)

    # Train Loop
    if args.do_train:
        train_df = pd.read_json(os.path.join(args.data_dir, 'train.json'))
        dev_df = pd.read_json(os.path.join(args.data_dir, 'dev.json'))
        
        train_df['comment'] = train_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        dev_df['comment'] = dev_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        
        train_dataset = TomBERTDataset(train_df, tokenizer, args.image_dir, roi_df, args.list_aspect, args.num_imgs, args.num_rois)
        dev_dataset = TomBERTDataset(dev_df, tokenizer, args.image_dir, roi_df, args.list_aspect, args.num_imgs, args.num_rois)
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        total_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
        logger.info(f"  Total optimization steps = {total_steps}")
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*total_steps), num_training_steps=total_steps)
        scaler = GradScaler() if args.fp16 else None
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0.0
        
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        
        for epoch in train_iterator:
            model.train(); resnet_img.train(); resnet_roi.train()
            optimizer.zero_grad()
            
            # Use 'tepoch' style for progress bar, matching FCMF
            with tqdm(train_loader, desc="Iteration", dynamic_ncols=True) as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    
                    t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                    t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                    tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                    sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                    labels = labels.to(device)
                    
                    # [FIX] Use torch.amp.autocast to avoid warnings
                    with torch.amp.autocast('cuda', enabled=args.fp16):
                        encoded_img = [resnet_img(t_img[:, i]).view(t_img.size(0), 2048, 49).permute(0, 2, 1) for i in range(t_img.shape[1])]
                        vis_embeds = torch.stack(encoded_img, dim=1)
                        
                        encoded_roi = []
                        for i in range(roi_img.shape[1]):
                            rois_in_img = [resnet_roi(roi_img[:, i, r]) for r in range(roi_img.shape[2])]
                            encoded_roi.append(torch.stack(rois_in_img, dim=1))
                        roi_embeds = torch.stack(encoded_roi, dim=1)
                        
                        loss = 0
                        for asp_idx in range(len(args.list_aspect)):
                            logits = model(tgt_ids[:, asp_idx], tgt_mask[:, asp_idx], sent_ids[:, asp_idx], sent_mask[:, asp_idx], vis_embeds, roi_embeds)
                            loss += criterion(logits, labels[:, asp_idx])
                        loss = loss / args.gradient_accumulation_steps
                    
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16: scaler.step(optimizer); scaler.update()
                        else: optimizer.step()
                        scheduler.step(); optimizer.zero_grad()
                    
                    tepoch.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)
            
            # End of Epoch
            logger.info(f"--> Epoch {epoch} Completed.")
            
            # Eval
            logger.info("***** Running evaluation on Dev Set *****")
            model.eval(); resnet_img.eval(); resnet_roi.eval()
            true_labels = {asp: [] for asp in args.list_aspect}
            pred_labels = {asp: [] for asp in args.list_aspect}
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Evaluating Dev", leave=False):
                    t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                    t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                    tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                    sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                    
                    encoded_img = [resnet_img(t_img[:, i]).view(t_img.size(0), 2048, 49).permute(0, 2, 1) for i in range(t_img.shape[1])]
                    vis_embeds = torch.stack(encoded_img, dim=1)
                    encoded_roi = []
                    for i in range(roi_img.shape[1]):
                        rois_in_img = [resnet_roi(roi_img[:, i, r]) for r in range(roi_img.shape[2])]
                        encoded_roi.append(torch.stack(rois_in_img, dim=1))
                    roi_embeds = torch.stack(encoded_roi, dim=1)
                    
                    for asp_idx, asp_name in enumerate(args.list_aspect):
                        logits = model(tgt_ids[:, asp_idx], tgt_mask[:, asp_idx], sent_ids[:, asp_idx], sent_mask[:, asp_idx], vis_embeds, roi_embeds)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        true_labels[asp_name].extend(labels[:, asp_idx].numpy())
                        pred_labels[asp_name].extend(preds)
            
            f1_scores = [macro_f1(true_labels[asp], pred_labels[asp])[2] for asp in args.list_aspect]
            avg_f1 = np.mean(f1_scores)
            logger.info(f"  Dev Macro-F1: {avg_f1}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                logger.info(f"  New Best F1 ({best_f1:.4f})! Saving best model...")
                save_model(os.path.join(args.output_dir, 'tombert_best.pth'), model, optimizer, scheduler, epoch, best_f1, scaler)

    # Test Loop
    if args.do_eval:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        best_path = os.path.join(args.output_dir, 'tombert_best.pth')
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("No best model found! Using current weights.")
            
        test_df = pd.read_json(os.path.join(args.data_dir, 'test.json'))
        test_df['comment'] = test_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        test_dataset = TomBERTDataset(test_df, tokenizer, args.image_dir, roi_df, args.list_aspect, args.num_imgs, args.num_rois)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        model.eval(); resnet_img.eval(); resnet_roi.eval()
        test_true = {asp: [] for asp in args.list_aspect}
        test_pred = {asp: [] for asp in args.list_aspect}
        formatted_results = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, batch_texts = batch
                t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                
                encoded_img = [resnet_img(t_img[:, i]).view(t_img.size(0), 2048, 49).permute(0, 2, 1) for i in range(t_img.shape[1])]
                vis_embeds = torch.stack(encoded_img, dim=1)
                encoded_roi = []
                for i in range(roi_img.shape[1]):
                    rois_in_img = [resnet_roi(roi_img[:, i, r]) for r in range(roi_img.shape[2])]
                    encoded_roi.append(torch.stack(rois_in_img, dim=1))
                roi_embeds = torch.stack(encoded_roi, dim=1)
                
                batch_logs = [{"text": t, "aspects": {}} for t in batch_texts]
                
                for asp_idx, asp_name in enumerate(args.list_aspect):
                    logits = model(tgt_ids[:, asp_idx], tgt_mask[:, asp_idx], sent_ids[:, asp_idx], sent_mask[:, asp_idx], vis_embeds, roi_embeds)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    test_true[asp_name].extend(labels[:, asp_idx].numpy())
                    test_pred[asp_name].extend(preds)
                    
                    for i, (p, l) in enumerate(zip(preds, labels[:, asp_idx].numpy())):
                        batch_logs[i]["aspects"][asp_name] = {"predict": POLARITY_MAP[p], "label": POLARITY_MAP[l]}
                
                formatted_results.extend(batch_logs)
        
        # Save results
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            f.write("***** Test results *****\n")
            all_f1 = 0
            for asp in args.list_aspect:
                p, r, f1 = macro_f1(test_true[asp], test_pred[asp])
                all_f1 += f1
                msg = f"{asp} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}"
                f.write(msg + "\n"); logger.info(msg)
            
            avg_f1 = all_f1 / len(args.list_aspect)
            f.write(f"Average F1: {avg_f1:.4f}\n"); logger.info(f"Average F1: {avg_f1:.4f}")

        # Formatted detailed log
        log_path = os.path.join(args.output_dir, "test_predictions_formatted.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"TEST DETAILED PREDICTIONS\nAverage Macro F1: {avg_f1:.4f}\n{'='*50}\n\n")
            for i, sample in enumerate(formatted_results):
                f.write("{\n")
                f.write(f"Sentence {i}: {sample['text']}\n")
                for asp in args.list_aspect:
                    res = sample['aspects'].get(asp, {'predict': 'N/A', 'label': 'N/A'})
                    f.write(f"   {asp}: Predict: {res['predict']}, Label: {res['label']}\n")
                f.write("}\n")
        logger.info(f"Formatted predictions saved to {log_path}")

if __name__ == '__main__':
    main()