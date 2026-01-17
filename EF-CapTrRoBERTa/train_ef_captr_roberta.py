import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from text_preprocess import TextNormalize, convert_unicode
from underthesea import text_normalize # Nếu cần dùng hàm text_normalize của underthesea như fcmf

# ==============================================================================
# 1. UTILS (COPY FROM FCMF)
# ==============================================================================

POLARITY_MAP = {0: 'None', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
    return p_macro, r_macro, f_macro

def save_model(path, model, optimizer, scheduler, epoch, best_score=0.0, scaler=None):
    if hasattr(model, 'module'): model_state = model.module.state_dict()
    else: model_state = model.state_dict()
    
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
# 2. DATASET (EF-CAPTR)
# ==============================================================================

class EFCapDataset(Dataset):
    def __init__(self, data, tokenizer, caption_dict, num_img, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.caption_dict = caption_dict
        self.num_img = num_img
        self.max_len = max_len
        
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
        self.pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['comment']
        img_paths = row['list_img']
        
        # --- Lấy Caption (Translation) ---
        captions = []
        for img_name in img_paths[:self.num_img]:
            # Thử tìm tên file trong dict caption
            cap = self.caption_dict.get(img_name)
            if not cap:
                 cap = self.caption_dict.get(os.path.basename(img_name))
            if cap: captions.append(cap)
            
        if not captions: caption_str = "hình ảnh bình thường"
        else: caption_str = ". ".join(captions)

        # --- Xử lý Label ---
        text_img_label = row['text_img_label']
        existing_aspects = {}
        for item in text_img_label:
            asp, pol = item.split("#")
            if "_" in asp: asp = "Public area"
            existing_aspects[asp] = pol

        input_ids_list, mask_list, label_list = [], [], []

        for asp in self.ASPECT:
            target_aspect = asp.replace('_', ' ')
            label_str = existing_aspects.get(asp, "None")
            label_id = self.pola_to_num[label_str]

            # [CLS] Review [SEP] Aspect . Caption [SEP]
            text_a = text 
            text_b = f"{target_aspect} . {caption_str}" 

            encodings = self.tokenizer(
                text_a, 
                text_b,
                max_length=self.max_len,
                padding='max_length',
                truncation='only_first',
                return_tensors='pt'
            )

            input_ids_list.append(encodings['input_ids'].squeeze(0))
            mask_list.append(encodings['attention_mask'].squeeze(0))
            label_list.append(label_id)

        return (torch.stack(input_ids_list), 
                torch.stack(mask_list), 
                torch.tensor(label_list, dtype=torch.long),
                text) # Trả về text gốc để log giống FCMF

# ==============================================================================
# 3. MODEL
# ==============================================================================

class EFCapTrRoBERTa(nn.Module):
    def __init__(self, pretrained_path, num_labels=4):
        super(EFCapTrRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_path)
        config = self.roberta.config
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None: module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # CLS token
        logits = self.classifier(self.dropout(pooled_output))
        return logits

# ==============================================================================
# 4. MAIN (UPDATED LOGGING STYLE)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../vimacsa', type=str)
    parser.add_argument("--caption_file", default='visual_captions_vi.json', type=str)
    parser.add_argument("--output_dir", default='./ef_captr_output', type=str)
    parser.add_argument("--pretrained_hf_model", default="xlm-roberta-base", type=str) # Đổi tên cho khớp fcmf
    
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--num_img", default=3, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Logging giống FCMF
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{args.output_dir}/training_ef_captr.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Xử lý batch size giống FCMF
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps")
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    normalizer = TextNormalize()

    # Load Caption
    cap_path = os.path.join(args.data_dir, args.caption_file)
    try:
        with open(cap_path, 'r', encoding='utf-8') as f: caption_dict = json.load(f)
        # Preprocess captions giống comments
        caption_dict = {k: normalizer.normalize(text_normalize(convert_unicode(v))) for k, v in caption_dict.items()}
    except:
        logger.warning(f"Caption file {cap_path} not found. Running without captions.")
        caption_dict = {}

    # Load Data
    if args.do_train:
        train_df = pd.read_json(os.path.join(args.data_dir, 'train.json'))
        dev_df = pd.read_json(os.path.join(args.data_dir, 'dev.json'))
        # Normalize text giống FCMF
        train_df['comment'] = train_df['comment'].apply(lambda x: normalizer.normalize(text_normalize(convert_unicode(x))))
        dev_df['comment'] = dev_df['comment'].apply(lambda x: normalizer.normalize(text_normalize(convert_unicode(x))))

        train_set = EFCapDataset(train_df, tokenizer, caption_dict, args.num_img, args.max_len)
        dev_set = EFCapDataset(dev_df, tokenizer, caption_dict, args.num_img, args.max_len)

    model = EFCapTrRoBERTa(args.pretrained_hf_model).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    num_train_steps = 0
    if args.do_train:
        num_train_steps = int(len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps * 0.1), num_training_steps=num_train_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if args.fp16 else None

    start_epoch = 0; max_f1 = 0.0
    
    # Resume Logic (Giống FCMF)
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Resuming from {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        max_f1 = ckpt.get('best_score', 0.0)
        if args.fp16 and 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])

    # --- TRAIN LOOP ---
    if args.do_train:
        train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.train_batch_size)
        dev_loader = DataLoader(dev_set, sampler=SequentialSampler(dev_set), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            logger.info(f"********** Epoch: {epoch} **********")
            model.train()
            optimizer.zero_grad()
            
            with tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True) as tepoch:
                for step, batch in enumerate(tepoch):
                    # Unpack (text ở cuối)
                    input_ids, attention_mask, labels, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    
                    with autocast(enabled=args.fp16):
                        loss = 0
                        for i in range(6): # 6 aspects
                            logits = model(input_ids[:,i,:], attention_mask[:,i,:])
                            loss += criterion(logits, labels[:,i])
                        
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16: scaler.scale(loss).backward()
                    else: loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer); scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        scheduler.step(); optimizer.zero_grad()
                    
                    tepoch.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

            # Eval
            if args.do_eval:
                model.eval()
                true_label, pred_label = {asp:[] for asp in train_set.ASPECT}, {asp:[] for asp in train_set.ASPECT}
                ASPECT = train_set.ASPECT
                
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Eval Dev", leave=False):
                        input_ids, attention_mask, labels, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                        for i, asp in enumerate(ASPECT):
                            logits = model(input_ids[:,i,:], attention_mask[:,i,:])
                            pred_label[asp].append(np.argmax(logits.cpu().numpy(), axis=-1))
                            true_label[asp].append(labels[:,i].cpu().numpy())

                total_f1 = 0
                for asp in ASPECT:
                    _, _, f1 = macro_f1(np.concatenate(true_label[asp]), np.concatenate(pred_label[asp]))
                    total_f1 += f1
                avg_f1 = total_f1 / len(ASPECT)
                logger.info(f"  Dev Macro F1: {avg_f1}")

                # Save Best Model (Format tên file giống FCMF)
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    logger.info(f"  New Best F1 ({max_f1})! Saving best model...")
                    save_model(f'{args.output_dir}/seed_{args.seed}_ef_captr_model_best.pth', model, optimizer, scheduler, epoch, max_f1, scaler)

                # Save Last Model (Giống FCMF)
                # save_model(f'{args.output_dir}/seed_{args.seed}_ef_captr_model_last.pth', model, optimizer, scheduler, epoch, max_f1, scaler)

    # --- TEST ---
    if args.do_eval:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        test_df = pd.read_json(os.path.join(args.data_dir, 'test.json'))
        test_df['comment'] = test_df['comment'].apply(lambda x: normalizer.normalize(text_normalize(convert_unicode(x))))
        test_set = EFCapDataset(test_df, tokenizer, caption_dict, args.num_img, args.max_len)
        test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=args.eval_batch_size)
        
        # Load Best Model
        best_path = f'{args.output_dir}/seed_{args.seed}_ef_captr_model_best.pth'
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        
        model.eval()
        true_label_list = {asp:[] for asp in test_set.ASPECT}
        pred_label_list = {asp:[] for asp in test_set.ASPECT}
        formatted_results = []
        ASPECT = test_set.ASPECT

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids, attention_mask, labels, batch_texts = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                batch_logs = [{"text": t, "aspects": {}} for t in batch_texts]
                
                for i, asp in enumerate(ASPECT):
                    logits = model(input_ids[:,i,:], attention_mask[:,i,:])
                    preds = np.argmax(logits.cpu().numpy(), axis=-1)
                    true_labels = labels[:,i].cpu().numpy()
                    
                    true_label_list[asp].append(true_labels)
                    pred_label_list[asp].append(preds)
                    
                    for idx, (p, l) in enumerate(zip(preds, true_labels)):
                        batch_logs[idx]["aspects"][asp] = {
                            "predict": POLARITY_MAP.get(p, "Unknown"),
                            "label": POLARITY_MAP.get(l, "Unknown")
                        }
                formatted_results.extend(batch_logs)

        # 1. Save Metrics (Giống format test_results_fcmf.txt)
        with open(os.path.join(args.output_dir, "test_results_ef_captr.txt"), "w") as writer:
            writer.write("***** Test results *****\n")
            all_f1 = 0
            for asp in ASPECT:
                tr = np.concatenate(true_label_list[asp])
                pr = np.concatenate(pred_label_list[asp])
                p, r, f1 = macro_f1(tr, pr)
                all_f1 += f1
                writer.write(f"{asp} - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}\n")
                logger.info(f"{asp} - F1: {f1:.4f}")
            avg_f1 = all_f1 / len(ASPECT)
            writer.write(f"Average F1: {avg_f1:.4f}\n")
            logger.info(f"Average F1: {avg_f1:.4f}")

        # 2. Save Detailed Predictions (Giống format test_predictions_formatted.txt của FCMF)
        with open(f"{args.output_dir}/test_predictions_formatted.txt", "w", encoding="utf-8") as f:
            f.write(f"TEST DETAILED PREDICTIONS\n")
            f.write(f"Average Macro F1: {avg_f1:.4f}\n")
            f.write("="*50 + "\n\n")
            
            for i, sample in enumerate(formatted_results):
                f.write("{\n")
                f.write(f"Sentence {i}: {sample['text']}\n")
                for asp in ASPECT:
                    res = sample['aspects'].get(asp, {'predict': 'N/A', 'label': 'N/A'})
                    # [QUAN TRỌNG] Format giống hệt FCMF:
                    f.write(f"{asp}:\n")
                    f.write(f"   predict: {res['predict']}\n")
                    f.write(f"   label:   {res['label']}\n")
                f.write("}\n")

if __name__ == "__main__":
    main()