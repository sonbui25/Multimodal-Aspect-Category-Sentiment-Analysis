# train_tomroberta_vimacsa_full.py
# Full implementation of Multimodal TomBERT for ViMACSA
# Based on Official TomBERT Architecture (1-Layer Cross, 1-Layer Fusion)

import os
import sys
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

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import precision_recall_fscore_support
from text_preprocess import TextNormalize, convert_unicode

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

POLARITY_MAP = {0: 'None', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

def macro_f1(y_true, y_pred):
    """Calculate Macro-F1 Score"""
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0.0
    )
    return p_macro, r_macro, f_macro

def save_model(path, model, optimizer, scheduler, epoch, best_score=0.0, scaler=None):
    """Save model checkpoint"""
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
    print(f"Model saved to {path}")

# ==============================================================================
# 2. DATASET CLASS (TomBERT Input Logic)
# ==============================================================================

class TomBERTDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, num_img=3, num_roi=7):
        self.data = data # Pandas DataFrame
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
        self.pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
        self.num_to_pola = {v: k for k, v in self.pola_to_num.items()}
        
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
        # Access row by integer location
        row = self.data.iloc[idx]
        text = str(row['comment'])
        list_img_path = row['list_img']
        
        # --- 1. IMAGE PROCESSING ---
        list_img_features = []
        global_roi_features = []

        # Load Global Images
        for img_path in list_img_path[:self.num_img]:
            full_path = os.path.join(self.img_folder, img_path)
            try:
                if os.path.exists(full_path):
                    one_image = read_image(full_path, mode=ImageReadMode.RGB)
                    img_tensor = self.transform(one_image).unsqueeze(0)
                else:
                    raise FileNotFoundError
            except:
                # Padding black image if error
                img_tensor = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224) # Placeholder for ROI extraction
            
            list_img_features.append(img_tensor)
            
            # Extract ROIs for this image
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
            
            # Pad ROIs to num_roi
            while len(list_roi_img) < self.num_roi:
                list_roi_img.append(np.zeros((3, 224, 224)))
                
            global_roi_features.append(list_roi_img)

        # Pad Images to num_img
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        for i in range(min(len(list_img_features), self.num_img)):
            t_img_features[i] = list_img_features[i]

        # Pad ROI Groups to num_img
        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        for i in range(min(len(global_roi_features), self.num_img)):
            roi_img_features[i] = np.array(global_roi_features[i])

        # --- 2. TEXT PROCESSING (TOMBERT INPUTS) ---
        # Get Ground Truth Labels
        text_img_label = row.get('text_img_label', []) + row.get('text_label', [])
        labels_map = {asp: 0 for asp in self.ASPECT} # Default None
        
        for item in text_img_label:
            if '#' in item:
                parts = item.split('#')
                if len(parts) == 2:
                    asp, pol = parts
                    # Normalize aspect name
                    if "_" in asp: asp = "Public_area"
                    if asp in labels_map:
                        labels_map[asp] = self.pola_to_num.get(pol, 0)

        # Create Inputs for 6 Aspects
        out_tgt_ids, out_tgt_mask = [], []
        out_sent_ids, out_sent_mask = [], []
        out_labels = []

        for aspect in self.ASPECT:
            # 2.1. Auxiliary Input (Target): [CLS] Aspect [SEP]
            # Used for Query in Cross-Attention
            target_text = aspect.replace('_', ' ').lower()
            enc_tgt = self.tokenizer(target_text, max_length=16, padding='max_length', truncation=True)
            
            # 2.2. Main Input (Sentence): [CLS] Review [SEP] Aspect [SEP]
            # Used for Context
            # Note: XLM-R uses <s> and </s>. Tokenizer handles this automatically with text_pair.
            # TomBERT puts review first, then aspect.
            clean_review = text.replace('_', ' ').lower()
            enc_sent = self.tokenizer(
                text=clean_review, 
                text_pair=target_text,
                max_length=256, 
                padding='max_length', 
                truncation=True
            )

            out_tgt_ids.append(torch.tensor(enc_tgt['input_ids']))
            out_tgt_mask.append(torch.tensor(enc_tgt['attention_mask']))
            out_sent_ids.append(torch.tensor(enc_sent['input_ids']))
            out_sent_mask.append(torch.tensor(enc_sent['attention_mask']))
            out_labels.append(labels_map[aspect])

        return (
            t_img_features, 
            torch.tensor(roi_img_features, dtype=torch.float32),
            torch.stack(out_tgt_ids), 
            torch.stack(out_tgt_mask),
            torch.stack(out_sent_ids), 
            torch.stack(out_sent_mask),
            torch.tensor(out_labels, dtype=torch.long), 
            text # Return raw text for logging
        )

# ==============================================================================
# 3. MODELS (Strict TomBERT Architecture)
# ==============================================================================

class BERTLikePooler(nn.Module):
    """Dense -> Tanh pooler for Cross-Attn output"""
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))

class TargetImageMatching(nn.Module):
    """1-Layer Cross Attention Block"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), 
            nn.GELU(), 
            nn.Linear(hidden_size * 4, hidden_size), 
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target_feats, image_feats):
        # target_feats: Query, image_feats: Key/Value
        attn_out, _ = self.mha(query=target_feats, key=image_feats, value=image_feats)
        h1 = self.norm1(target_feats + self.dropout(attn_out))
        h2 = self.norm2(h1 + self.feed_forward(h1))
        return h2

class TomBERT(nn.Module):
    def __init__(self, pretrained_path, num_labels=4):
        super(TomBERT, self).__init__()
        # Load Pretrained
        self.roberta = AutoModel.from_pretrained(pretrained_path)
        # Aux BERT (Initialized from same weights for fairness)
        self.s2_roberta = AutoModel.from_pretrained(pretrained_path) 
        
        config = self.roberta.config
        self.hidden_size = config.hidden_size
        
        # Projections
        self.vis_projection = nn.Linear(2048, self.hidden_size)
        self.roi_projection = nn.Linear(2048, self.hidden_size)
        
        # 1. Cross-Encoder (1 Layer)
        self.ti_matching = nn.ModuleList([
            TargetImageMatching(self.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob) 
            for _ in range(1)
        ])
        
        # 2. Pooler for Cross-Output
        self.ent2img_pooler = BERTLikePooler(self.hidden_size)
        
        # 3. Multimodal Encoder (1 Layer)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=config.num_attention_heads, 
            dim_feedforward=config.intermediate_size, 
            dropout=config.hidden_dropout_prob, 
            activation="gelu", 
            batch_first=True
        )
        self.mm_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Init custom layers
        self._init_custom_weights()

    def _init_custom_weights(self):
        for module in [self.vis_projection, self.roi_projection, self.classifier, self.ti_matching, self.mm_encoder, self.ent2img_pooler]:
            for p in module.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, target_ids, target_mask, sentence_ids, sentence_mask, visual_embeds_att, roi_embeds_att):
        # Main Branch (Sentence)
        s_out = self.roberta(sentence_ids, attention_mask=sentence_mask)
        h_s = s_out.last_hidden_state
        
        # Aux Branch (Target/Aspect)
        t_out = self.s2_roberta(target_ids, attention_mask=target_mask)
        h_t = t_out.last_hidden_state
        
        # Visual Processing
        B, N_Img, Patches, Dim = visual_embeds_att.shape
        _, _, N_Roi, _ = roi_embeds_att.shape
        vis_flat = visual_embeds_att.view(B, N_Img * Patches, Dim)
        roi_flat = roi_embeds_att.view(B, N_Img * N_Roi, Dim)
        
        vis_proj = self.vis_projection(vis_flat)
        roi_proj = self.roi_projection(roi_flat)
        g_visual = torch.cat([vis_proj, roi_proj], dim=1)
        
        # Cross Attention
        h_v = h_t
        for layer in self.ti_matching:
            h_v = layer(target_feats=h_v, image_feats=g_visual)
            
        # Pooling Cross Output
        h_v_pooled = self.ent2img_pooler(h_v).unsqueeze(1) # [B, 1, H]
        
        # Fusion Input
        mm_input = torch.cat([h_v_pooled, h_s], dim=1)
        
        # Masking
        bsz = sentence_mask.size(0)
        valid_cls = torch.ones(bsz, 1).to(sentence_mask.device)
        mm_mask = torch.cat([valid_cls, sentence_mask], dim=1)
        src_key_padding_mask = (mm_mask == 0)
        
        # Fusion
        h_mm = self.mm_encoder(mm_input, src_key_padding_mask=src_key_padding_mask)
        
        # Final Pooling ('first' token)
        out_fused = h_mm[:, 0, :]
        logits = self.classifier(self.dropout(out_fused))
        return logits

# ResNet Wrappers (Inline to keep single file)
class myResNetImg(nn.Module):
    def __init__(self, resnet, if_fine_tune):
        super().__init__()
        self.resnet = resnet; self.if_fine_tune = if_fine_tune
    def forward(self, x):
        x = self.resnet.conv1(x); x = self.resnet.bn1(x); x = self.resnet.relu(x); x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.resnet.layer2(x); x = self.resnet.layer3(x); x = self.resnet.layer4(x)
        # Adaptive Pool to 7x7 -> 49 patches
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
        # Mean pool for ROI
        fc = x.mean(3).mean(2)
        if not self.if_fine_tune: fc = Variable(fc.data)
        return fc

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--image_dir', default='../vimacsa/image')
    parser.add_argument("--pretrained_hf_model", default="xlm-roberta-base", type=str)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fine_tune_cnn', action='store_true')
    
    args = parser.parse_args()
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Random Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    # Load Helper Objects
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    normalize_class = TextNormalize()
    
    # Load ROI Data
    roi_path = os.path.join(args.data_dir, "roi_data.csv")
    if os.path.exists(roi_path):
        roi_df = pd.read_csv(roi_path)
        # Ensure file_name includes extension if missing (adjust based on your csv)
        # roi_df['file_name'] = roi_df['file_name'].apply(lambda x: x if x.endswith('.png') else x + '.png')
    else:
        print("Warning: roi_data.csv not found. ROIs will be empty.")
        roi_df = pd.DataFrame(columns=['file_name', 'x1', 'y1', 'x2', 'y2'])

    # --- MODEL BUILD ---
    print("Building TomBERT Model...")
    model = TomBERT(args.pretrained_hf_model).to(device)
    
    # Visual Backbone
    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(img_res_model, args.fine_tune_cnn).to(device)
    resnet_roi = myResNetRoI(roi_res_model, args.fine_tune_cnn).to(device)

    # --- DATA & TRAINING ---
    ASPECTS = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
    
    if args.do_train:
        train_df = pd.read_json(os.path.join(args.data_dir, 'train.json'))
        dev_df = pd.read_json(os.path.join(args.data_dir, 'dev.json'))
        
        # Normalize Text
        train_df['comment'] = train_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        dev_df['comment'] = dev_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        
        train_dataset = TomBERTDataset(train_df, tokenizer, args.image_dir, roi_df)
        dev_dataset = TomBERTDataset(dev_df, tokenizer, args.image_dir, roi_df)
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        scaler = GradScaler() if args.fp16 else None
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0.0
        
        print("Starting Training...")
        for epoch in range(int(args.num_train_epochs)):
            model.train(); resnet_img.train(); resnet_roi.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                # Unpack Batch
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                
                # Move to device
                t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                labels = labels.to(device)
                
                with autocast(enabled=args.fp16):
                    # Visual Forward
                    # Img: [B, N, 3, 224, 224] -> [B, N, 2048, 49]
                    encoded_img = []
                    for i in range(t_img.shape[1]):
                        # [B, 3, H, W] -> [B, 2048, 7, 7] -> [B, 2048, 49] -> [B, 49, 2048]
                        feat = resnet_img(t_img[:, i])
                        feat = feat.view(feat.size(0), 2048, 49).permute(0, 2, 1)
                        encoded_img.append(feat)
                    vis_embeds = torch.stack(encoded_img, dim=1) # [B, N_Img, 49, 2048]
                    
                    # ROI: [B, N, R, 3, H, W]
                    encoded_roi = []
                    for i in range(roi_img.shape[1]):
                        rois_in_img = []
                        for r in range(roi_img.shape[2]):
                            # [B, 3, H, W] -> [B, 2048, 1, 1] -> [B, 2048]
                            feat = resnet_roi(roi_img[:, i, r])
                            rois_in_img.append(feat)
                        encoded_roi.append(torch.stack(rois_in_img, dim=1))
                    roi_embeds = torch.stack(encoded_roi, dim=1) # [B, N_Img, N_Roi, 2048]
                    
                    # Text Forward & Loss
                    loss = 0
                    for asp_idx in range(6): # 6 Aspects
                        logits = model(
                            target_ids=tgt_ids[:, asp_idx], target_mask=tgt_mask[:, asp_idx],
                            sentence_ids=sent_ids[:, asp_idx], sentence_mask=sent_mask[:, asp_idx],
                            visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds
                        )
                        loss += criterion(logits, labels[:, asp_idx])
                    
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                train_loss += loss.item() * args.gradient_accumulation_steps
                pbar.set_postfix({'loss': train_loss / (step+1)})
            
            # --- EVALUATION ON DEV ---
            print("Evaluating on Dev Set...")
            model.eval(); resnet_img.eval(); resnet_roi.eval()
            true_labels = {asp: [] for asp in ASPECTS}
            pred_labels = {asp: [] for asp in ASPECTS}
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, leave=False):
                    t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                    
                    t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                    tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                    sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                    
                    # Visual Feature Extraction (Repeated for Eval)
                    encoded_img = []
                    for i in range(t_img.shape[1]):
                        feat = resnet_img(t_img[:, i])
                        feat = feat.view(feat.size(0), 2048, 49).permute(0, 2, 1)
                        encoded_img.append(feat)
                    vis_embeds = torch.stack(encoded_img, dim=1)
                    
                    encoded_roi = []
                    for i in range(roi_img.shape[1]):
                        rois_in_img = []
                        for r in range(roi_img.shape[2]):
                            feat = resnet_roi(roi_img[:, i, r])
                            rois_in_img.append(feat)
                        encoded_roi.append(torch.stack(rois_in_img, dim=1))
                    roi_embeds = torch.stack(encoded_roi, dim=1)
                    
                    for asp_idx, asp_name in enumerate(ASPECTS):
                        logits = model(
                            target_ids=tgt_ids[:, asp_idx], target_mask=tgt_mask[:, asp_idx],
                            sentence_ids=sent_ids[:, asp_idx], sentence_mask=sent_mask[:, asp_idx],
                            visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds
                        )
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        true_labels[asp_name].extend(labels[:, asp_idx].numpy())
                        pred_labels[asp_name].extend(preds)
            
            # Calculate Macro F1
            f1_scores = []
            for asp in ASPECTS:
                _, _, f1 = macro_f1(true_labels[asp], pred_labels[asp])
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            print(f"Epoch {epoch+1} - Dev Macro F1: {avg_f1:.4f}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                save_model(os.path.join(args.output_dir, 'best_model.pth'), model, optimizer, scheduler, epoch, best_f1, scaler)

    # ==========================================================================
    # 5. TEST LOOP (Fully Implemented)
    # ==========================================================================
    if args.do_eval:
        print("\n" + "="*30)
        print("STARTING TEST EVALUATION")
        print("="*30)
        
        # Load Best Model
        best_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_path):
            print(f"Loading best checkpoint from {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: Best model not found, using current weights.")
            
        # Load Test Data
        test_df = pd.read_json(os.path.join(args.data_dir, 'test.json'))
        test_df['comment'] = test_df['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        test_dataset = TomBERTDataset(test_df, tokenizer, args.image_dir, roi_df)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        
        model.eval(); resnet_img.eval(); resnet_roi.eval()
        
        test_true = {asp: [] for asp in ASPECTS}
        test_pred = {asp: [] for asp in ASPECTS}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                
                t_img = t_img.to(device); roi_img = roi_img.to(device).float()
                tgt_ids = tgt_ids.to(device); tgt_mask = tgt_mask.to(device)
                sent_ids = sent_ids.to(device); sent_mask = sent_mask.to(device)
                
                # Visual Features
                encoded_img = []
                for i in range(t_img.shape[1]):
                    feat = resnet_img(t_img[:, i])
                    feat = feat.view(feat.size(0), 2048, 49).permute(0, 2, 1)
                    encoded_img.append(feat)
                vis_embeds = torch.stack(encoded_img, dim=1)
                
                encoded_roi = []
                for i in range(roi_img.shape[1]):
                    rois_in_img = []
                    for r in range(roi_img.shape[2]):
                        feat = resnet_roi(roi_img[:, i, r])
                        rois_in_img.append(feat)
                    encoded_roi.append(torch.stack(rois_in_img, dim=1))
                roi_embeds = torch.stack(encoded_roi, dim=1)
                
                for asp_idx, asp_name in enumerate(ASPECTS):
                    logits = model(
                        target_ids=tgt_ids[:, asp_idx], target_mask=tgt_mask[:, asp_idx],
                        sentence_ids=sent_ids[:, asp_idx], sentence_mask=sent_mask[:, asp_idx],
                        visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds
                    )
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    test_true[asp_name].extend(labels[:, asp_idx].numpy())
                    test_pred[asp_name].extend(preds)
        
        # Calculate & Save Results
        result_str = "TEST RESULTS (Multimodal TomBERT Baseline)\n" + "-"*40 + "\n"
        avg_f1 = 0
        for asp in ASPECTS:
            p, r, f1 = macro_f1(test_true[asp], test_pred[asp])
            avg_f1 += f1
            result_str += f"{asp:<15} | F1: {f1:.4f} | P: {p:.4f} | R: {r:.4f}\n"
        
        avg_f1 /= len(ASPECTS)
        result_str += "-"*40 + f"\nAVERAGE MACRO F1: {avg_f1:.4f}\n"
        
        print(result_str)
        
        with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
            f.write(result_str)
            
        print(f"Test results saved to {os.path.join(args.output_dir, 'test_results.txt')}")

if __name__ == '__main__':
    main()