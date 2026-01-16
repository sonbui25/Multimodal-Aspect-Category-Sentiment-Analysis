import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet152, ResNet152_Weights

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from text_preprocess import TextNormalize, convert_unicode

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

        for img_path in list_img_path[:self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            try:
                one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except:
                img_transform = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224)
            list_img_features.append(img_transform)
            
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
                for _ in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
            
            global_roi_features.append(list_roi_img)

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

            target_text = asp.lower()
            tokenized_tgt = self.tokenizer(target_text, max_length=16, padding='max_length', truncation=True)
            
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
# 3. MODELS (TomBERT)
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
        if not self.if_fine_tune: att = att.detach()
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
        if not self.if_fine_tune: fc = fc.detach()
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
        ff_out = self.feed_forward(h_v)
        h_v = self.norm2(h_v + ff_out)
        return h_v

class TomBERT(nn.Module):
    def __init__(self, pretrained_path, num_labels=4):
        super(TomBERT, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_path)
        config = self.roberta.config
        self.hidden_size = config.hidden_size
        
        self.vis_projection = nn.Linear(2048, self.hidden_size)
        self.roi_projection = nn.Linear(2048, self.hidden_size)
        
        self.ti_matching = nn.ModuleList([TargetImageMatching(self.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob) for _ in range(1)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob, activation="gelu", batch_first=True)
        self.mm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels)
        
        self.apply_custom_init(self.classifier); self.apply_custom_init(self.vis_projection); self.apply_custom_init(self.roi_projection)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding): module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): module.bias.data.zero_(); module.weight.data.fill_(1.0)
    def apply_custom_init(self, module): module.apply(self._init_weights)

    def forward(self, target_ids, target_mask, sentence_ids, sentence_mask, visual_embeds_att, roi_embeds_att):
        # --- BATCH FOLDING FIX FOR DDP ---
        # Input shapes: 
        # target_ids: (B, Num_Asp, Seq_Len)
        # visual_embeds_att: (B, Num_Img, 49, 2048)
        
        B, Num_Asp, Seq_Len = target_ids.shape
        
        # 1. Flatten Aspect Dimension: (B * Num_Asp, Seq_Len)
        flat_target_ids = target_ids.view(-1, Seq_Len)
        flat_target_mask = target_mask.view(-1, Seq_Len)
        flat_sentence_ids = sentence_ids.view(-1, 170) # Max len 170 defined in dataset
        flat_sentence_mask = sentence_mask.view(-1, 170)
        
        # 2. Flatten & Repeat Image Features
        # visual_embeds: (B, Num_Img, 49, 2048) -> need (B * Num_Asp, Num_Img, 49, 2048)
        flat_visual_embeds = visual_embeds_att.repeat_interleave(Num_Asp, dim=0)
        flat_roi_embeds = roi_embeds_att.repeat_interleave(Num_Asp, dim=0)
        
        # 3. Forward Pass (Vectorized)
        t_out = self.roberta(flat_target_ids, attention_mask=flat_target_mask)
        h_t = t_out.last_hidden_state
        s_out = self.roberta(flat_sentence_ids, attention_mask=flat_sentence_mask)
        h_s = s_out.last_hidden_state

        B_flat, N_Img, Patches, Dim = flat_visual_embeds.shape
        _, _, N_Roi, _ = flat_roi_embeds.shape
        
        vis_flat = flat_visual_embeds.view(B_flat, N_Img * Patches, Dim)
        roi_flat = flat_roi_embeds.view(B_flat, N_Img * N_Roi, Dim)
        
        vis_proj = self.vis_projection(vis_flat)
        roi_proj = self.roi_projection(roi_flat)
        g_visual = torch.cat([vis_proj, roi_proj], dim=1)
        
        h_v = h_t
        for layer in self.ti_matching: 
            h_v = layer(target_feats=h_v, image_feats=g_visual)
        
        # Multimodal Encoder
        h_v_cls = h_v[:, 0:1, :]
        mm_input = torch.cat([h_v_cls, h_s], dim=1)
        
        # Masking
        valid_cls = torch.ones(B_flat, 1).to(flat_sentence_mask.device)
        mm_mask = torch.cat([valid_cls, flat_sentence_mask], dim=1)
        src_key_padding_mask = (mm_mask == 0) 
        
        h_mm = self.mm_encoder(mm_input, src_key_padding_mask=src_key_padding_mask)
        
        out_vis = h_mm[:, 0, :]
        out_txt = h_mm[:, 1, :]
        pooled_output = torch.cat([out_vis, out_txt], dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        
        # Logits: (B * Num_Asp, Num_Classes)
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
    parser.add_argument("--num_imgs", default=7, type=int)
    parser.add_argument("--num_rois", default=4, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=13, type=float)
    parser.add_argument("--warmup_proportion", default=0.0, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fine_tune_cnn', action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ddp", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    args = parser.parse_args()
    
    # --- DDP SETUP ---
    if args.ddp:
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    print(f"Running on device: {ddp_local_rank if args.ddp else device}")
    if args.ddp and ddp_world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
    
    os.makedirs(args.output_dir, exist_ok=True)
    if master_process:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, handlers=[logging.FileHandler(f'{args.output_dir}/training_tombert.log'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    # [MATCHING FCMF LOGIC] Batch Size Calculation
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter")
    
    # Chia batch size cho accumulation steps giống FCMF
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    normalize_class = TextNormalize()
    ASPECT = args.list_aspect

    roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv"); roi_df['file_name'] = roi_df['file_name'] + '.png'
    with open(f'{args.data_dir}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
    with open(f'{args.data_dir}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)

    # --- Load Data ---
    if args.do_train:
        train_data = pd.read_json(f'{args.data_dir}/train.json')
        dev_data = pd.read_json(f'{args.data_dir}/dev.json')
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        
        train_dataset = TomBERTDataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        dev_dataset = TomBERTDataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)

    # --- Models ---
    model = TomBERT(pretrained_path=args.pretrained_hf_model, num_labels=args.num_polarity)
    model.to(device)
    
    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(resnet=img_res_model, if_fine_tune=args.fine_tune_cnn).to(device)
    resnet_roi = myResNetRoI(resnet=roi_res_model, if_fine_tune=args.fine_tune_cnn).to(device)
    
    if args.ddp and ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        resnet_img = DDP(resnet_img, device_ids=[ddp_local_rank], find_unused_parameters=True)
        resnet_roi = DDP(resnet_roi, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # --- Optimizer ---
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler() if args.fp16 else None
    
    num_train_steps = 0
    if args.do_train:
        num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps * args.warmup_proportion), num_training_steps=num_train_steps)

    start_epoch = 0; max_f1 = 0.0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Resuming from {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict']); scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1; max_f1 = ckpt.get('best_score', 0.0)

    # --- TRAINING LOOP ---
    if args.do_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.ddp else RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            if args.ddp: train_sampler.set_epoch(epoch)
            logger.info(f"********** Epoch: {epoch} **********")
            model.train(); resnet_img.train(); resnet_roi.train()
            optimizer.zero_grad()
            
            with tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, disable=not master_process) as tepoch:
                for step, batch in enumerate(tepoch):
                    batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                    t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                    roi_img = roi_img.float()

                    encoded_img = []
                    for img_idx in range(args.num_imgs):
                        img_f = resnet_img(t_img[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                        encoded_img.append(img_f)
                    encoded_roi = []
                    for img_idx in range(args.num_imgs):
                        roi_list = [resnet_roi(roi_img[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                        encoded_roi.append(torch.stack(roi_list, dim=1))
                    
                    vis_embeds = torch.stack(encoded_img, dim=1)
                    roi_embeds = torch.stack(encoded_roi, dim=1)

                    with autocast(enabled=args.fp16):
                        # [FIXED LOOP] Pass EVERYTHING at once
                        # Model internally flattens (Batch * Aspects)
                        logits = model(target_ids=tgt_ids, target_mask=tgt_mask,
                                       sentence_ids=sent_ids, sentence_mask=sent_mask,
                                       visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                        
                        # Labels: (B, Num_Asp) -> (B * Num_Asp)
                        flat_labels = labels.view(-1)
                        loss = criterion(logits, flat_labels)

                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                    if args.fp16: scaler.scale(loss).backward()
                    else: loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16: scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if args.fp16: scaler.step(optimizer); scaler.update()
                        else: optimizer.step()
                        scheduler.step(); optimizer.zero_grad()
                    
                    tepoch.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

            # --- Eval ---
            if args.do_eval and master_process:
                model.eval(); resnet_img.eval(); resnet_roi.eval()
                true_label, pred_label = [], []
                
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Evaluating Dev", leave=False):
                        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                        t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels, _ = batch
                        roi_img = roi_img.float()
                        
                        # Feature Extraction
                        encoded_img = []
                        for img_idx in range(args.num_imgs):
                            img_f = resnet_img(t_img[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                            encoded_img.append(img_f)
                        encoded_roi = []
                        for img_idx in range(args.num_imgs):
                            roi_list = [resnet_roi(roi_img[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                            encoded_roi.append(torch.stack(roi_list, dim=1))
                        vis_embeds = torch.stack(encoded_img, dim=1)
                        roi_embeds = torch.stack(encoded_roi, dim=1)

                        # Vectorized Forward
                        logits = model(target_ids=tgt_ids, target_mask=tgt_mask,
                                       sentence_ids=sent_ids, sentence_mask=sent_mask,
                                       visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                        
                        preds = np.argmax(logits.cpu().numpy(), axis=-1)
                        lbls = labels.view(-1).cpu().numpy()
                        
                        true_label.extend(lbls)
                        pred_label.extend(preds)

                # Calculate Macro F1 (Flattened)
                _, _, avg_f1 = macro_f1(true_label, pred_label)
                logger.info(f"  Dev Macro F1: {avg_f1}")
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    save_model(f'{args.output_dir}/tombert_best.pth', model, optimizer, scheduler, epoch, max_f1, scaler)

    # --- TEST ---
    if args.do_eval and master_process:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        
        # 1. Load Test Data
        test_data = pd.read_json(f'{args.data_dir}/test.json')
        test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(convert_unicode(x)))
        test_dataset = TomBERTDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

        # 2. Load Best Model Checkpoint
        best_path = f'{args.output_dir}/tombert_best.pth'
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            # Handle DDP loading (unwrap wrapper if needed)
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("No best model found! Using current weights from end of training.")

        model.eval(); resnet_img.eval(); resnet_roi.eval()

        true_label_list = {asp:[] for asp in ASPECT}
        pred_label_list = {asp:[] for asp in ASPECT}
        formatted_results = [] # To store detailed logs per sentence

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Extract raw text for logging (last element in batch)
                batch_texts = batch[-1]
                
                # Move tensors to device
                batch_tensors = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch[:-1])
                t_img, roi_img, tgt_ids, tgt_mask, sent_ids, sent_mask, labels = batch_tensors
                roi_img = roi_img.float()

                # --- Feature Extraction (Same as Train) ---
                encoded_img = []
                for img_idx in range(args.num_imgs):
                    img_f = resnet_img(t_img[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                    encoded_img.append(img_f)
                
                encoded_roi = []
                for img_idx in range(args.num_imgs):
                    roi_list = [resnet_roi(roi_img[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                    encoded_roi.append(torch.stack(roi_list, dim=1))
                
                vis_embeds = torch.stack(encoded_img, dim=1)
                roi_embeds = torch.stack(encoded_roi, dim=1)

                # --- Vectorized Forward Pass ---
                # Model inputs are flattened internally or we pass them directly 
                # (Note: The corrected TomBERT model provided handles flattening internally inside forward)
                logits = model(target_ids=tgt_ids, target_mask=tgt_mask,
                               sentence_ids=sent_ids, sentence_mask=sent_mask,
                               visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds)
                
                # logits shape is (Batch * Num_Aspects, Num_Classes)
                # We need to reshape back to (Batch, Num_Aspects, Num_Classes) to log per sentence
                batch_size_curr = t_img.size(0)
                num_aspects = len(ASPECT)
                logits = logits.view(batch_size_curr, num_aspects, -1)
                
                # Get Predictions & Labels
                preds = np.argmax(logits.cpu().numpy(), axis=-1)   # Shape: [Batch, Num_Aspects]
                true_labels = labels.cpu().numpy()                 # Shape: [Batch, Num_Aspects]

                # --- Collect Results ---
                # Initialize log entry for each sentence in batch
                batch_logs = [{"text": t, "aspects": {}} for t in batch_texts]

                for aspect_idx, aspect_name in enumerate(ASPECT):
                    # 1. Aggregate for Metrics Calculation
                    true_label_list[aspect_name].append(true_labels[:, aspect_idx])
                    pred_label_list[aspect_name].append(preds[:, aspect_idx])
                    
                    # 2. Detailed Logging (Group by Sentence)
                    for sample_idx in range(batch_size_curr):
                        p = preds[sample_idx, aspect_idx]
                        l = true_labels[sample_idx, aspect_idx]
                        batch_logs[sample_idx]["aspects"][aspect_name] = {
                            "predict": POLARITY_MAP.get(p, "Unknown"),
                            "label": POLARITY_MAP.get(l, "Unknown")
                        }
                
                formatted_results.extend(batch_logs)

        # 3. Calculate & Save Metrics (Identical format to FCMF)
        output_eval_file = os.path.join(args.output_dir, "test_results_tombert.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Test results *****\n")
            all_f1 = 0
            for asp in ASPECT:
                tr = np.concatenate(true_label_list[asp])
                pr = np.concatenate(pred_label_list[asp])
                precision, recall, f1 = macro_f1(tr, pr)
                all_f1 += f1
                writer.write(f"{asp} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}\n")
                logger.info(f"{asp} - F1: {f1:.4f}")
            
            avg_f1 = all_f1 / len(ASPECT)
            writer.write(f"Average F1: {avg_f1:.4f}\n")
            logger.info(f"Average F1: {avg_f1:.4f}")

        # 4. Save Formatted Detailed Log (Identical format to FCMF)
        log_path = f"{args.output_dir}/test_predictions_formatted.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"TEST DETAILED PREDICTIONS\n")
            f.write(f"Average Macro F1: {avg_f1:.4f}\n")
            f.write("="*50 + "\n\n")
            
            for i, sample in enumerate(formatted_results):
                f.write("{\n")
                f.write(f"Sentence {i}: {sample['text']}\n")
                for asp in ASPECT:
                    res = sample['aspects'].get(asp, {'predict': 'N/A', 'label': 'N/A'})
                    f.write(f"{asp}:\n")
                    f.write(f"   predict: {res['predict']}\n")
                    f.write(f"   label:   {res['label']}\n")
                f.write("}\n")
        
        logger.info(f"Formatted predictions saved to {log_path}")

if __name__ == '__main__':
    main()