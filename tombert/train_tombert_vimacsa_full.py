import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet152, ResNet152_Weights

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from ..text_preprocess import TextNormalize, convert_unicode

# ==============================================================================
# 1. UTILS
# ==============================================================================

POLARITY_MAP = {0: 'None', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
    return p_macro, r_macro, f_macro

def save_model(path, model, optimizer, scheduler, epoch, best_score=0.0, scaler=None):
    if hasattr(model, 'module'): model_state = model.module.state_dict()
    else: model_state = model.state_dict()
    checkpoint_dict = {
        "epoch": epoch, "best_score": best_score, "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler is not None: checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint_dict, path)

# ==============================================================================
# 2. DATASET (TomBERT Specific)
# ==============================================================================

class TomBERTDataset(Dataset):
    """
    Dataset class dành riêng cho TomBERT.
    Khác với FCMF, TomBERT cần tách biệt 'Target Input' và 'Sentence Input' 
    để đưa vào 2 encoder riêng biệt trước khi fusion.
    """
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img, num_roi):
        self.data = data
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
        self.pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
        self.transform = transforms.Compose([
                            transforms.Resize((224,224),antialias=True),  
                            transforms.ConvertImageDtype(torch.float32),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])
        self.roi_df = roi_df
        self.img_folder = img_folder
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        idx_data = self.data.iloc[idx, :].values
        text = idx_data[0]
        
        # --- Image Loading ---
        list_img_path = idx_data[1]
        list_img_features, global_roi_features = [], []

        # Load Full Images
        for img_path in list_img_path[:self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            try:
                one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except:
                img_transform = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224)
            list_img_features.append(img_transform)
            
            # Load ROIs
            roi_in_img_df = self.roi_df[self.roi_df['file_name'] == img_path][:self.num_roi]
            list_roi_img = []
            
            if roi_in_img_df.shape[0] == 0:
                list_roi_img = np.zeros((self.num_roi, 3, 224, 224))
            else:
                for i_roi in range(roi_in_img_df.shape[0]):
                    x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1:5].values
                    roi_in_image = one_image[:, x1:x2, y1:y2]
                    if roi_in_image.numel() == 0: roi_transform = torch.zeros(3, 224, 224).numpy()
                    else: roi_transform = self.transform(roi_in_image).numpy()
                    list_roi_img.append(roi_transform)
                # Padding ROIs if not enough
                for _ in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
            
            global_roi_features.append(list_roi_img)

        # Padding Images if not enough
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        for i in range(min(len(list_img_features), self.num_img)):
            t_img_features[i,:] = list_img_features[i]

        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        if len(global_roi_features) > 0:
            for i in range(min(len(global_roi_features), self.num_img)):
                roi_img_features[i,:] = np.asarray(global_roi_features[i])

        # --- Text Processing ---
        text_img_label = idx_data[3]
        list_aspect, list_polar = [], []
        for asp_pol in text_img_label:
            asp, pol = asp_pol.split("#")
            if "_" in asp: asp = "Public area"
            list_aspect.append(asp)
            list_polar.append(pol)

        for asp in self.ASPECT:
            if "_" in asp: asp = "Public area"
            if asp not in list_aspect:
                list_aspect.append(asp)
                list_polar.append('None')

        out_tgt_ids, out_tgt_mask = [], []
        out_sent_ids, out_sent_mask = [], []
        out_labels = []

        for ix in range(len(self.ASPECT)):
            asp = self.ASPECT[ix]
            if "_" in asp: asp = "Public area"
            idx_asp_in_list_asp = list_aspect.index(asp)

            # 1. Target Input: Aspect Only (e.g., "Service")
            target_text = asp.lower()
            tokenized_tgt = self.tokenizer(target_text, max_length=16, padding='max_length', truncation=True)
            
            # 2. Sentence Input: Aspect [SEP] Text (Standard ABSA format)
            sentence_text = f"{asp} </s></s> {text}".lower().replace('_', ' ')
            tokenized_sent = self.tokenizer(sentence_text, max_length=170, padding='max_length', truncation=True)

            out_tgt_ids.append(torch.tensor(tokenized_tgt['input_ids']))
            out_tgt_mask.append(torch.tensor(tokenized_tgt['attention_mask']))
            out_sent_ids.append(torch.tensor(tokenized_sent['input_ids']))
            out_sent_mask.append(torch.tensor(tokenized_sent['attention_mask']))
            out_labels.append(self.pola_to_num[list_polar[idx_asp_in_list_asp]])

        return t_img_features, torch.tensor(roi_img_features), \
               torch.stack(out_tgt_ids), torch.stack(out_tgt_mask), \
               torch.stack(out_sent_ids), torch.stack(out_sent_mask), \
               torch.tensor(out_labels), text

# ==============================================================================
# 3. MODELS (TomBERT Best Version)
# ==============================================================================

class myResNetImg(nn.Module):
    def __init__(self, resnet, if_fine_tune):
        super(myResNetImg, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])
        if not self.if_fine_tune: att = Variable(att.data)
        return att

class myResNetRoI(nn.Module):
    def __init__(self, resnet, if_fine_tune):
        super(myResNetRoI, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
    def forward(self, x):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        fc = x.mean(3).mean(2)
        if not self.if_fine_tune: fc = Variable(fc.data)
        return fc

class TargetImageMatching(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(TargetImageMatching, self).__init__()
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
        config = self.roberta.config
        self.hidden_size = config.hidden_size
        
        self.vis_projection = nn.Linear(2048, self.hidden_size)
        self.roi_projection = nn.Linear(2048, self.hidden_size)
        
        # Best Config: L_t=5, L_m=4
        self.ti_matching = nn.ModuleList([TargetImageMatching(self.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob) for _ in range(5)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob, activation="gelu", batch_first=True)
        self.mm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels) # *2 for BOTH pooling
        
        self.apply_custom_init(self.classifier); self.apply_custom_init(self.vis_projection); self.apply_custom_init(self.roi_projection)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding): module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): module.bias.data.zero_(); module.weight.data.fill_(1.0)
    def apply_custom_init(self, module): module.apply(self._init_weights)

    def forward(self, target_ids, target_mask, sentence_ids, sentence_mask, visual_embeds_att, roi_embeds_att):
        t_out = self.roberta(target_ids, attention_mask=target_mask)
        h_t = t_out.last_hidden_state
        s_out = self.roberta(sentence_ids, attention_mask=sentence_mask)
        h_s = s_out.last_hidden_state

        B, N_Img, Patches, Dim = visual_embeds_att.shape; _, _, N_Roi, _ = roi_embeds_att.shape
        vis_flat = visual_embeds_att.view(B, N_Img * Patches, Dim); roi_flat = roi_embeds_att.view(B, N_Img * N_Roi, Dim)
        vis_proj = self.vis_projection(vis_flat); roi_proj = self.roi_projection(roi_flat)
        g_visual = torch.cat([vis_proj, roi_proj], dim=1)
        
        h_v = h_t
        for layer in self.ti_matching: h_v = layer(target_feats=h_v, image_feats=g_visual)
        
        # Multimodal Encoder with First-Text Concatenation
        h_v_cls = h_v[:, 0:1, :]
        mm_input = torch.cat([h_v_cls, h_s], dim=1)
        
        bsz = sentence_mask.size(0)
        valid_cls = torch.ones(bsz, 1).to(sentence_mask.device)
        mm_mask = torch.cat([valid_cls, sentence_mask], dim=1)
        src_key_padding_mask = (mm_mask == 0) 
        
        h_mm = self.mm_encoder(mm_input, src_key_padding_mask=src_key_padding_mask)
        
        # BOTH Pooling
        out_vis = h_mm[:, 0, :]
        out_txt = h_mm[:, 1, :]
        pooled_output = torch.cat([out_vis, out_txt], dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        return logits

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--image_dir', default='../vimacsa/image', help='path to images')
    parser.add_argument("--pretrained_hf_model", default="xlm-roberta-base", type=str)
    parser.add_argument("--list_aspect", default=['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area'], nargs='+')
    parser.add_argument("--num_polarity", default=4, type=int)
    parser.add_argument("--num_imgs", default=3, type=int)
    parser.add_argument("--num_rois", default=7, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fine_tune_cnn', action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    args = parser.parse_args()

    if args.no_cuda: device = torch.device("cpu")
    else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, handlers=[logging.FileHandler(f'{args.output_dir}/training_tombert.log'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    normalize_class = TextNormalize() # Sử dụng từ module import
    ASPECT = args.list_aspect

    roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv"); roi_df['file_name'] = roi_df['file_name'] + '.png'
    with open(f'{args.data_dir}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
    with open(f'{args.data_dir}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)

    # --- Load Data ---
    if args.do_train:
        train_data = pd.read_json(f'{args.data_dir}/train.json')
        dev_data = pd.read_json(f'{args.data_dir}/dev.json')
        # Sử dụng hàm từ text_preprocess
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        
        train_dataset = TomBERTDataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        dev_dataset = TomBERTDataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)

    # --- Setup Models ---
    model = TomBERT(pretrained_path=args.pretrained_hf_model, num_labels=args.num_polarity)
    model.to(device)
    
    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(resnet=img_res_model, if_fine_tune=args.fine_tune_cnn).to(device)
    resnet_roi = myResNetRoI(resnet=roi_res_model, if_fine_tune=args.fine_tune_cnn).to(device)

    # --- Optimizer ---
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler() if args.fp16 else None
    
    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs) if args.do_train else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps * args.warmup_proportion), num_training_steps=num_train_steps)

    start_epoch = 0; max_f1 = 0.0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Resuming from {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict']); scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1; max_f1 = ckpt.get('best_score', 0.0)

    # --- TRAINING ---
    if args.do_train:
        train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
        dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            model.train(); resnet_img.train(); resnet_roi.train()
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                roi_img = roi_img.float()

                with autocast(enabled=args.fp16):
                    encoded_img = [resnet_img(t_img[:,i,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) for i in range(args.num_imgs)]
                    encoded_roi = [torch.stack([resnet_roi(roi_img[:,i,r,:]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                    vis_embeds = torch.stack(encoded_img, dim=1); roi_embeds = torch.stack(encoded_roi, dim=1)

                    loss = 0
                    for id_asp in range(len(ASPECT)):
                        logits = model(target_ids=tgt_ids[:,id_asp,:], target_mask=tgt_mask[:,id_asp,:],
                                       sentence_ids=sent_ids[:,id_asp,:], sentence_mask=sent_mask[:,id_asp,:],
                                       visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                        loss += criterion(logits, labels[:,id_asp])
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16: scaler.scale(loss).backward()
                else: loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                    else: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    scheduler.step(); optimizer.zero_grad()

            if args.do_eval:
                model.eval()
                true_label, pred_label = {asp:[] for asp in ASPECT}, {asp:[] for asp in ASPECT}
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Dev Eval"):
                        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                        t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                        roi_img = roi_img.float()
                        encoded_img = [resnet_img(t_img[:,i,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) for i in range(args.num_imgs)]
                        encoded_roi = [torch.stack([resnet_roi(roi_img[:,i,r,:]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(encoded_img, dim=1); roi_embeds = torch.stack(encoded_roi, dim=1)

                        for id_asp in range(len(ASPECT)):
                            logits = model(target_ids=tgt_ids[:,id_asp,:], target_mask=tgt_mask[:,id_asp,:],
                                           sentence_ids=sent_ids[:,id_asp,:], sentence_mask=sent_mask[:,id_asp,:],
                                           visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                            pred_label[ASPECT[id_asp]].append(np.argmax(logits.cpu().numpy(), axis=-1))
                            true_label[ASPECT[id_asp]].append(labels[:,id_asp].cpu().numpy())

                total_f1 = 0
                for asp in ASPECT:
                    _, _, f1 = macro_f1(np.concatenate(true_label[asp]), np.concatenate(pred_label[asp]))
                    total_f1 += f1
                avg_f1 = total_f1 / len(ASPECT)
                logger.info(f"Dev Macro F1: {avg_f1}")
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    save_model(f'{args.output_dir}/tombert_best.pth', model, optimizer, scheduler, epoch, max_f1, scaler)

    # --- TEST ---
    if args.do_eval:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        test_data = pd.read_json(f'{args.data_dir}/test.json')
        test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        test_dataset = TomBERTDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

        best_path = f'{args.output_dir}/tombert_best.pth'
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else: logger.warning("No best model found! Using current weights.")

        model.eval(); resnet_img.eval(); resnet_roi.eval()
        true_label_list = {asp:[] for asp in ASPECT}
        pred_label_list = {asp:[] for asp in ASPECT}
        formatted_results = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch_texts = batch[-1]
                batch_tensors = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch[:-1])
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels = batch_tensors
                roi_img = roi_img.float()

                encoded_img = [resnet_img(t_img[:,i,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) for i in range(args.num_imgs)]
                encoded_roi = [torch.stack([resnet_roi(roi_img[:,i,r,:]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                vis_embeds = torch.stack(encoded_img, dim=1); roi_embeds = torch.stack(encoded_roi, dim=1)

                batch_logs = [{"text": t, "aspects": {}} for t in batch_texts]

                for id_asp in range(len(ASPECT)):
                    logits = model(target_ids=tgt_ids[:,id_asp,:], target_mask=tgt_mask[:,id_asp,:],
                                   sentence_ids=sent_ids[:,id_asp,:], sentence_mask=sent_mask[:,id_asp,:],
                                   visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                    preds = np.argmax(logits.cpu().numpy(), axis=-1)
                    true_labels = labels[:,id_asp].cpu().numpy()
                    
                    true_label_list[ASPECT[id_asp]].append(true_labels)
                    pred_label_list[ASPECT[id_asp]].append(preds)

                    for i, (p, l) in enumerate(zip(preds, true_labels)):
                        batch_logs[i]["aspects"][ASPECT[id_asp]] = {"predict": POLARITY_MAP.get(p, "Unknown"), "label": POLARITY_MAP.get(l, "Unknown")}
                
                formatted_results.extend(batch_logs)

        # Save Metrics
        with open(os.path.join(args.output_dir, "test_results_tombert.txt"), "w") as writer:
            writer.write("***** Test results *****\n")
            all_f1 = 0
            for asp in ASPECT:
                tr = np.concatenate(true_label_list[asp])
                pr = np.concatenate(pred_label_list[asp])
                p, r, f1 = macro_f1(tr, pr)
                all_f1 += f1
                writer.write(f"{asp} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
            avg_f1 = all_f1 / len(ASPECT)
            writer.write(f"Average F1: {avg_f1:.4f}\n")
            logger.info(f"Test Average F1: {avg_f1:.4f}")

        # Save Logs
        with open(f"{args.output_dir}/test_predictions_formatted.txt", "w", encoding="utf-8") as f:
            f.write(f"TEST DETAILED PREDICTIONS\nAverage Macro F1: {avg_f1:.4f}\n{'='*50}\n\n")
            for i, sample in enumerate(formatted_results):
                f.write("{\n")
                f.write(f"Sentence {i}: {sample['text']}\n")
                for asp in ASPECT:
                    res = sample['aspects'].get(asp, {'predict': 'N/A', 'label': 'N/A'})
                    f.write(f"   {asp}: Predict: {res['predict']}, Label: {res['label']}\n")
                f.write("}\n")
        logger.info(f"Saved predictions to {args.output_dir}")

if __name__ == '__main__':
    main()