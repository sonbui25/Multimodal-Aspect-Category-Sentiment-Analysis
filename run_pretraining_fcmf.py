import torch
import torch.nn.functional as F
import math
from text_preprocess import *
from iaog_dataset import IAOGDataset 
from fcmf_framework.fcmf_pretraining import FCMFSeq2Seq, beam_search
import argparse
import logging
import random
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from underthesea import text_normalize
from fcmf_framework.resnet_utils import *
from torchvision.models import resnet152, ResNet152_Weights
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json
from torch.cuda.amp import autocast 
import os
from rouge_score import rouge_scorer

def save_model(path, model, optimizer, scheduler, epoch, best_score):
    if hasattr(model, 'module'): model_state = model.module.state_dict()
    else: model_state = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score, 
    }, path)
def calculate_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
def main():
    parser = argparse.ArgumentParser()
    # --- ARGUMENTS ---
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True)
    parser.add_argument("--pretrained_data_dir", default='../iaog-pretraining', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--pretrained_hf_model", default=None, type=str, required=True)
    parser.add_argument('--image_dir', default='../vimacsa/image')
    parser.add_argument('--resnet_label_path', default='/kaggle/input/resnet-output')
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)
    
    parser.add_argument("--max_seq_length", default=170, type=int)
    parser.add_argument("--max_len_decoder", default=20, type=int)
    parser.add_argument("--num_imgs", default=7, type=int)
    parser.add_argument("--num_rois", default=4, type=int)
    parser.add_argument('--fine_tune_cnn', action='store_true')
    parser.add_argument("--alpha", default=0.8, type=float)
    parser.add_argument("--beam_size", default=2, type=int)
    
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=8.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ddp", action='store_true')
    parser.add_argument("--list_aspect", nargs='+', default=[]) 

    args = parser.parse_args()

    # --- 1. SETUP ---
    if args.no_cuda: device = torch.device("cpu"); ddp_local_rank = 0; master_process = True
    elif args.ddp:
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(ddp_local_rank)
        dist.init_process_group(backend='nccl')
        device = f'cuda:{ddp_local_rank}'; master_process = ddp_local_rank == 0
        ddp_world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ddp_local_rank = 0; master_process = True; ddp_world_size = 1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO if master_process else logging.WARN)
    logger = logging.getLogger(__name__)
    if master_process: os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if ddp_world_size > 1: torch.cuda.manual_seed_all(args.seed)

    # --- 2. DATA PREP ---
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<iaog>']})
    normalize_class = TextNormalize()

    try:
        roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv")
        roi_df['file_name'] = roi_df['file_name'] + '.png'
        json_path = args.resnet_label_path if os.path.exists(f'{args.resnet_label_path}/resnet152_image_label.json') else args.data_dir
        with open(f'{json_path}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
        with open(f'{json_path}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)
    except: return

    ASPECT_LIST = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']

    if args.do_train:
        train_data = pd.read_json(f'{args.pretrained_data_dir}/train_with_iaog.json')
        dev_data = pd.read_json(f'{args.pretrained_data_dir}/dev_with_iaog.json')
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        
        if ddp_world_size > 1:
            chunk = len(train_data) // ddp_world_size
            train_data = train_data.iloc[chunk*ddp_local_rank : chunk*(ddp_local_rank+1)]

        train_dataset = IAOGDataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder)
        dev_dataset = IAOGDataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder)

    # --- 3. MODEL ---
    model = FCMFSeq2Seq(len(tokenizer), args.max_len_decoder, args.pretrained_hf_model, args.num_imgs, args.num_rois, args.alpha)
    model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))
    model.decoder.embedding = torch.nn.Embedding(len(tokenizer), model.decoder.num_hiddens)
    
    img_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(img_res, args.fine_tune_cnn, device).to(device)
    resnet_roi = myResNetRoI(roi_res, args.fine_tune_cnn, device).to(device)
    model = model.to(device)

    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        resnet_img = DDP(resnet_img, device_ids=[ddp_local_rank])
        resnet_roi = DDP(resnet_roi, device_ids=[ddp_local_rank])

    # --- 4. OPTIMIZER ---
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    num_train_steps = len(train_dataset) // args.train_batch_size * args.num_train_epochs if args.do_train else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps*args.warmup_proportion, num_training_steps=num_train_steps)

    start_epoch = 0
    max_f1_score = 0.0 # [CHANGED] Đổi thành max_f1_score

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict']) if not isinstance(model, DDP) else model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'best_score' in ckpt: max_rougeL = ckpt['best_score']
        logger.info(f"Resumed from epoch {start_epoch}, Best ROUGE-L: {max_rougeL}")

    # Rouge Scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # --- 5. TRAINING LOOP ---
    if args.do_train:
        train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset) if args.ddp else RandomSampler(train_dataset), batch_size=args.train_batch_size)
        dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            if args.ddp: train_loader.sampler.set_epoch(epoch)
            model.train(); resnet_img.train(); resnet_roi.train()
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, disable=not master_process)
            for step, batch in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch}")
                
                batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                (t_img_f, roi_img_f, roi_coors, all_labels, all_dec_ids, all_dec_mask, 
                 all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _, _) = batch 
                
                roi_img_f = roi_img_f.float()

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    with torch.no_grad():
                        enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        roi_embeds = torch.stack(enc_rois, dim=1)

                    # --- LOOP QUA 6 ASPECT ---
                    # ==============================================================================
                    # [FIXED STRATEGY] SAMPLE-WISE WEIGHTING
                    # Sentiment Samples: Weight = 1.0 (Học mạnh)
                    # None Samples:      Weight = NONE_SAMPLE_WEIGHT (Học nhẹ để tránh spam, giữ negative constraint)
                    # ==============================================================================
                    
                    # Định nghĩa trọng số cho mẫu None (Tune số này: 0.1, 0.15, 0.2)
                    NONE_SAMPLE_WEIGHT = 0.05
                    
                    total_loss = 0
                    
                    for id_asp in range(6):
                        enc_ids = all_enc_ids[:, id_asp, :]
                        enc_type = all_enc_type[:, id_asp, :]
                        enc_mask = all_enc_mask[:, id_asp, :]
                        add_mask = all_add_mask[:, id_asp, :]
                        
                        dec_ids = all_dec_ids[:, id_asp, :]
                        labels = all_labels[:, id_asp, :]

                        # 1. Xác định mẫu nào là 'none', mẫu nào là 'sentiment'
                        # Decode label ra text để kiểm tra chính xác tuyệt đối
                        # (Lưu ý: labels đang là tensor trên GPU, cần chuyển về cpu numpy để decode)
                        lbl_ids = labels.detach().cpu().numpy()
                        lbl_ids = np.where(lbl_ids != -100, lbl_ids, tokenizer.pad_token_id)
                        decoded_texts = tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)
                        
                        # Tạo mask trọng số: 1.0 cho sentiment, NONE_SAMPLE_WEIGHT cho none
                        batch_sample_weights = []
                        for txt in decoded_texts:
                            clean_txt = txt.lower().strip()
                            if "none" in clean_txt or clean_txt == "":
                                batch_sample_weights.append(NONE_SAMPLE_WEIGHT)
                            else:
                                batch_sample_weights.append(1.0)
                        
                        # Chuyển thành tensor đưa lên GPU
                        sample_weights = torch.tensor(batch_sample_weights, device=device) # Shape: [Batch]

                        # 2. Forward Model
                        logits = model(enc_X=enc_ids, dec_X=dec_ids, 
                                    visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                    token_type_ids=enc_type, attention_mask=enc_mask, added_attention_mask=add_mask, is_train=True)
                        
                        # 3. Tính Loss riêng cho từng mẫu (reduction='none')
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                        
                        # Logits: [Batch, Seq, Vocab] -> Permute: [Batch, Vocab, Seq]
                        loss_per_token = loss_fct(logits.permute(0, 2, 1), labels) # -> [Batch, Seq]
                        loss_per_sample = loss_per_token.sum(dim=1) # -> [Batch] (Tổng loss của câu)
                        
                        # 4. Áp dụng trọng số
                        weighted_loss = loss_per_sample * sample_weights
                        
                        # 5. Tính trung bình (Chia cho tổng trọng số để loss không bị nhỏ đi một cách giả tạo)
                        # Hoặc đơn giản là mean() nếu bạn muốn loss phản ánh đúng tỷ lệ đóng góp
                        loss_asp = weighted_loss.mean() 
                        
                        total_loss += loss_asp

                # Scale và Backward như cũ...
                if args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / args.gradient_accumulation_steps

                if args.fp16: scaler.scale(total_loss).backward()
                else: total_loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.fp16: scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    scheduler.step(); optimizer.zero_grad()
                    pbar.set_postfix(loss=total_loss.item() * args.gradient_accumulation_steps)

            # --- EVALUATION ---
            if master_process and args.do_eval:
                logger.info("***** Running evaluation on Dev Set with BEAM SEARCH *****")
                model.eval(); resnet_img.eval(); resnet_roi.eval()
                
                total_val_loss = 0
                # val_rouge1_scores = []
                # val_rouge2_scores = [] # [ADDED] List chứa điểm Rouge-2
                # val_rougeL_scores = [] # [ADDED] List chứa điểm Rouge-L
                total_TP = 0
                total_FP = 0
                total_FN = 0
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Eval Beam", leave=False):
                        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                        (t_img_f, roi_img_f, roi_coors, all_labels, all_dec_ids, _, 
                         all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _, _) = batch
                        
                        enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        roi_embeds = torch.stack(enc_rois, dim=1)

                        for id_asp in range(6):
                            # # --- A. TÍNH LOSS (Teacher Forcing) ---
                            # logits_tf = model(enc_X=all_enc_ids[:,id_asp], dec_X=all_dec_ids[:,id_asp],
                            #                visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                            #                token_type_ids=all_enc_type[:,id_asp], attention_mask=all_enc_mask[:,id_asp], 
                            #                added_attention_mask=all_add_mask[:,id_asp], is_train=True)
                            # loss_item = criterion(logits_tf.reshape(-1, logits_tf.size(-1)), all_labels[:,id_asp].reshape(-1)).item()
                            # total_val_loss += loss_item

                            # --- B. SINH TEXT BẰNG BEAM SEARCH ---
                            batch_size = all_enc_ids.shape[0]
                            decoded_preds = []
                            
                            for i in range(batch_size):
                                pred = beam_search(
                                    model=model,
                                    tokenizer=tokenizer,
                                    enc_ids=all_enc_ids[i, id_asp],
                                    enc_mask=all_enc_mask[i, id_asp],
                                    enc_type=all_enc_type[i, id_asp],
                                    add_mask=all_add_mask[i, id_asp],
                                    vis_embeds=vis_embeds[i],
                                    roi_embeds=roi_embeds[i],
                                    roi_coors=roi_coors[i],
                                    beam_size=args.beam_size,
                                    max_len=args.max_len_decoder,
                                    device=device
                                )[0]
                                decoded_preds.append(pred)

                            # --- C. TÍNH ROUGE L ---
                            lbls = all_labels[:,id_asp].cpu().numpy()
                            lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
                            decoded_lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
                            def parse_sentiment_string(text):
                                if not text or text.lower().strip() == 'none':
                                    return set()
                                return set(s.strip() for s in text.lower().split(','))
                            
                            for p_text, g_text in zip(decoded_preds, decoded_lbls):
                                # [ADDED] Tiền xử lý predict: Nếu có "n " ở đầu (từ Beam Search), xóa đi
                                if p_text.startswith("n ") and len(p_text) > 2: p_text = p_text[2:]
                                
                                # [ADDED] Phân tích chuỗi thành tập hợp các từ cảm xúc
                                pred_set = parse_sentiment_string(p_text)
                                gold_set = parse_sentiment_string(g_text)
                                
                                # [ADDED] Tính TP, FP, FN cho MẪU NÀY
                                TP = len(pred_set.intersection(gold_set))
                                FP = len(pred_set.difference(gold_set))
                                FN = len(gold_set.difference(pred_set))
                                
                                # [ADDED] Cập nhật tổng
                                total_TP += TP
                                total_FP += FP
                                total_FN += FN
                                # val_rouge1_scores.append(scores['rouge1'].fmeasure)
                                # val_rouge2_scores.append(scores['rouge2'].fmeasure) # [ADDED]
                                # val_rougeL_scores.append(scores['rougeL'].fmeasure) # [ADDED]

                # avg_val_loss = total_val_loss / (len(dev_loader) * 6)
                # avg_val_rouge1 = np.mean(val_rouge1_scores) if val_rouge1_scores else 0.0
                # avg_val_rouge2 = np.mean(val_rouge2_scores) if val_rouge2_scores else 0.0 # [ADDED]
                # avg_val_rougeL = np.mean(val_rougeL_scores) if val_rougeL_scores else 0.0 # [ADDED]
                avg_val_P, avg_val_R, avg_val_F1 = calculate_f1(total_TP, total_FP, total_FN)
                # [CHANGED] Log cả 3 chỉ số
                logger.info(f"Epoch {epoch} | P: {avg_val_P:.4f} | R: {avg_val_R:.4f} | F1: {avg_val_F1:.4f}")

                # [CHANGED] Tối ưu hóa dựa trên ROUGE-L
                if avg_val_F1 > max_f1_score:
                    max_f1_score = avg_val_F1
                    logger.info(f"New Best F1-Score ({max_f1_score:.4f})! Saving model...")
                    save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth', model, optimizer, scheduler, epoch, best_score=max_f1_score)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_best.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_f1_score)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_best.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_f1_score)
                    
                save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_last.pth', model, optimizer, scheduler, epoch, best_score=max_f1_score)
                save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_last.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_f1_score)
                save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_last.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_f1_score)
                print("\n")

    # --- 6. TEST (WITH BEAM SEARCH & FULL ROUGE) ---
    if args.do_eval and master_process:
        try:
            test_data = pd.read_json(f'{args.pretrained_data_dir}/test_with_iaog.json')
            test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
            test_loader = DataLoader(IAOGDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder), batch_size=args.eval_batch_size)
        except: return

        ckpt_path = f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(model, DDP): model.module.load_state_dict(ckpt['model_state_dict'])
            else: model.load_state_dict(ckpt['model_state_dict'])
        
        model.eval(); resnet_img.eval(); resnet_roi.eval()
        
        all_test_results = []
        # all_rouge1 = []
        # all_rouge2 = [] # [ADDED]
        # all_rougeL = []
        test_TP = 0
        test_FP = 0
        test_FN = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test with Beam Search"):
                batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                
                (t_img_f, roi_img_f, roi_coors, all_dec_lbls, _, _, 
                 all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _, batch_texts) = batch

                enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                vis_embeds = torch.stack(enc_imgs, dim=1)
                enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                roi_embeds = torch.stack(enc_rois, dim=1)

                batch_results = [{"text": t, "aspects": {}} for t in batch_texts]
                batch_size = all_enc_ids.shape[0]

                for id_asp in range(6):
                    aspect_name = ASPECT_LIST[id_asp]
                    decoded_preds = []

                    for i in range(batch_size):
                        pred_text = beam_search(
                            model=model,
                            tokenizer=tokenizer,
                            enc_ids=all_enc_ids[i, id_asp],
                            enc_mask=all_enc_mask[i, id_asp],
                            enc_type=all_enc_type[i, id_asp],
                            add_mask=all_add_mask[i, id_asp],
                            vis_embeds=vis_embeds[i],
                            roi_embeds=roi_embeds[i],
                            roi_coors=roi_coors[i],
                            beam_size=args.beam_size,
                            max_len=args.max_len_decoder,
                            device=device
                        )[0]
                        decoded_preds.append(pred_text)

                    lbls = all_dec_lbls[:,id_asp].cpu().numpy()
                    lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
                    decoded_lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
                    
                    for i, (p, g) in enumerate(zip(decoded_preds, decoded_lbls)):
                        if p.startswith("n ") and len(p) > 2: p = p[2:]
                        
                        # [ADDED] Phân tích chuỗi thành tập hợp các từ cảm xúc
                        pred_set = parse_sentiment_string(p)
                        gold_set = parse_sentiment_string(g)
                        
                        # [ADDED] Tính TP, FP, FN cho MẪU NÀY
                        TP = len(pred_set.intersection(gold_set))
                        FP = len(pred_set.difference(gold_set))
                        FN = len(gold_set.difference(pred_set))
                        
                        # [ADDED] Cập nhật tổng
                        test_TP += TP
                        test_FP += FP
                        test_FN += FN
                        
                        batch_results[i]["aspects"][aspect_name] = {"predict": p, "label": g}
                        # all_rouge1.append(scores['rouge1'].fmeasure)
                        # all_rouge2.append(scores['rouge2'].fmeasure) # [ADDED]
                        # all_rougeL.append(scores['rougeL'].fmeasure)
                
                all_test_results.extend(batch_results)

        # avg_r1 = np.mean(all_rouge1)
        # avg_r2 = np.mean(all_rouge2) # [ADDED]
        # avg_rL = np.mean(all_rougeL)
        avg_rP, avg_rR, avg_rF1 = calculate_f1(test_TP, test_FP, test_FN)
        logger.info(f"***** TEST RESULTS *****")
        # logger.info(f"Test ROUGE-1: {avg_r1:.4f}")
        # logger.info(f"Test ROUGE-2: {avg_r2:.4f}")
        # logger.info(f"Test ROUGE-L: {avg_rL:.4f}")
        logger.info(f"Test Precision: {avg_rP:.4f}") # [CHANGED]
        logger.info(f"Test Recall:    {avg_rR:.4f}") # [CHANGED]
        logger.info(f"Test F1-Score:  {avg_rF1:.4f}") # [CHANGED]
        log_path = f"{args.output_dir}/test_predictions_formatted.txt"
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"TEST METRICS:\n")
            # f.write(f"ROUGE-1: {avg_r1:.4f}\n")
            # f.write(f"ROUGE-2: {avg_r2:.4f}\n")
            # f.write(f"ROUGE-L: {avg_rL:.4f}\n")
            f.write(f"Precision: {avg_rP:.4f}\n")
            f.write(f"Recall: {avg_rR:.4f}\n")
            f.write(f"F1-Score: {avg_rF1:.4f}\n")
            f.write("="*50 + "\n\n")
            
            for i, sample in enumerate(all_test_results):
                f.write("{\n")
                f.write(f"Sentence {i}: {sample['text']}\n")
                for asp in ASPECT_LIST:
                    res = sample['aspects'].get(asp, {'predict': 'N/A', 'label': 'N/A'})
                    f.write(f"{asp}:\n")
                    f.write(f"   predict: {res['predict']}\n")
                    f.write(f"   label:   {res['label']}\n")
                f.write("}\n")
        
        logger.info(f"Formatted predictions saved to {log_path}")

if __name__ == '__main__':
    main()