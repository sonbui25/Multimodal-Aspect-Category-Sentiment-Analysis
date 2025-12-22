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
from bert_score import score

def save_model(path, model, optimizer, scheduler, epoch, best_score, scaler=None):
    if hasattr(model, 'module'): model_state = model.module.state_dict()
    else: model_state = model.state_dict()
    
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score, 
    }
    
    if scaler is not None:
        checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint_dict, path)

def main():
    parser = argparse.ArgumentParser()
    # --- ARGUMENTS ---
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True)
    parser.add_argument("--pretrained_data_dir", default='../iaog-pretraining', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--pretrained_hf_model", default=None, type=str, required=True)
    
    # Argument cho BERTScore (như chỉnh sửa của bạn)
    parser.add_argument('--bert_score_model', default='uitnlp/visobert', type=str, 
                        help="HuggingFace model name or local path for BERTScore")
    
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
        logger.info(f"ROI DataFrame loaded with {len(roi_df)} entries.")
    except:
        raise ValueError("Can't find roi_data.csv")
    
    try:
        with open(f'{args.data_dir}/resnet152_image_label.json') as imf:
            dict_image_aspect = json.load(imf)
            logger.info(f"Image aspect categories loaded with {len(dict_image_aspect)} entries.")

        with open(f'{args.data_dir}/resnet152_roi_label.json') as rf:
            dict_roi_aspect = json.load(rf)
            logger.info(f"ROI aspect categories loaded with {len(dict_roi_aspect)} entries.")
    except:
        raise ValueError("Get image/roi aspect category first. Please run run_image_categories.py or run_roi_categories.py")

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
    
    if args.fp16:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    num_train_steps = 0
    if args.do_train:
        num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps*args.warmup_proportion), num_training_steps=num_train_steps)

    start_epoch = 0
    max_f1_score = 0.0 

    # A. RESUME (Pretraining)
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if os.path.isfile(checkpoint_path):
            if master_process: logger.info(f"--> Resuming from checkpoint: {checkpoint_path}")
            
            # Load file checkpoint
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 1. Load Model Weights
            if isinstance(model, (DDP, torch.nn.DataParallel)): 
                model.module.load_state_dict(ckpt['model_state_dict'])
            else: 
                model.load_state_dict(ckpt['model_state_dict'])
            
            # 2. Load ResNet
            resimg_path = checkpoint_path.replace("iaog_model", "resimg_model")
            resroi_path = checkpoint_path.replace("iaog_model", "resroi_model")
            
            if os.path.exists(resimg_path):
                if master_process: logger.info(f"    Loading ResNet Img: {resimg_path}")
                unwrap_resimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
                unwrap_resimg.load_state_dict(torch.load(resimg_path, map_location=device, weights_only=False)['model_state_dict'])
            
            if os.path.exists(resroi_path):
                if master_process: logger.info(f"    Loading ResNet RoI: {resroi_path}")
                unwrap_resroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
                unwrap_resroi.load_state_dict(torch.load(resroi_path, map_location=device, weights_only=False)['model_state_dict'])

            # 3. Load Optimizer
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            # 4. Load Scheduler
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if master_process: logger.info("--> Scheduler state loaded successfully.")

            # 5. Load Scaler
            if args.fp16 and 'scaler_state_dict' in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            
            # 6. Thiết lập epoch bắt đầu
            start_epoch = ckpt['epoch'] + 1
            if 'best_score' in ckpt: max_f1_score = ckpt['best_score']
            
            if master_process: 
                logger.info(f"Resumed at epoch {start_epoch}, Best F1: {max_f1_score}")
                # Kiểm tra LR thực tế sau khi load
                logger.info(f"Current LR after resume: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            if master_process: logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
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

                with torch.amp.autocast('cuda', enabled=args.fp16):
                    with torch.no_grad():
                        enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        roi_embeds = torch.stack(enc_rois, dim=1)

                    NONE_SAMPLE_WEIGHT = 0.1
                    total_loss = 0
                    
                    for id_asp in range(6):
                        enc_ids = all_enc_ids[:, id_asp, :]
                        enc_type = all_enc_type[:, id_asp, :]
                        enc_mask = all_enc_mask[:, id_asp, :]
                        add_mask = all_add_mask[:, id_asp, :]
                        
                        dec_ids = all_dec_ids[:, id_asp, :]
                        labels = all_labels[:, id_asp, :]

                        lbl_ids = labels.detach().cpu().numpy()
                        lbl_ids = np.where(lbl_ids != -100, lbl_ids, tokenizer.pad_token_id)
                        decoded_texts = tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)
                        
                        batch_sample_weights = []
                        for txt in decoded_texts:
                            clean_txt = txt.lower().strip()
                            if "none" in clean_txt or clean_txt == "":
                                batch_sample_weights.append(NONE_SAMPLE_WEIGHT)
                            else:
                                batch_sample_weights.append(1.0)
                        
                        sample_weights = torch.tensor(batch_sample_weights, device=device) 

                        logits = model(enc_X=enc_ids, dec_X=dec_ids, 
                                    visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                    token_type_ids=enc_type, attention_mask=enc_mask, added_attention_mask=add_mask, is_train=True)
                        
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                        loss_per_token = loss_fct(logits.permute(0, 2, 1), labels) 
                        loss_per_sample = loss_per_token.sum(dim=1) 
                        weighted_loss = loss_per_sample * sample_weights
                        loss_asp = weighted_loss.mean() 
                        total_loss += loss_asp

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
                
                # --- LOGIC TƯƠNG TỰ RUN_MULTIMODAL: Lưu theo Dictionary ---
                val_preds = {asp: [] for asp in ASPECT_LIST}
                val_refs = {asp: [] for asp in ASPECT_LIST}
                
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
                            aspect_name = ASPECT_LIST[id_asp]
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

                            lbls = all_labels[:,id_asp].cpu().numpy()
                            lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
                            decoded_lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
                            
                            for p_text, g_text in zip(decoded_preds, decoded_lbls):
                                if p_text.startswith("n ") and len(p_text) > 2: p_text = p_text[2:]
                                
                                val_preds[aspect_name].append(p_text)
                                val_refs[aspect_name].append(g_text)

                logger.info(f"Computing BERTScore for Dev Set using model: {args.bert_score_model} ...")
                logger.info("***** BERTScore Per Aspect *****")
                
                total_P, total_R, total_F1 = 0, 0, 0
                
                # --- LOGIC TƯƠNG TỰ RUN_MULTIMODAL: Tính Score từng Aspect rồi cộng dồn ---
                for asp in ASPECT_LIST:
                    P, R, F1 = score(val_preds[asp], val_refs[asp], lang='vi', model_type=args.bert_score_model, verbose=False, device=device, num_layers=12)
                    
                    asp_P = P.mean().item()
                    asp_R = R.mean().item()
                    asp_F1 = F1.mean().item()
                    
                    logger.info(f"  Aspect: {asp:<15} | P: {asp_P:.4f} | R: {asp_R:.4f} | F1: {asp_F1:.4f}")
                    
                    total_P += asp_P
                    total_R += asp_R
                    total_F1 += asp_F1
                
                avg_val_P = total_P / len(ASPECT_LIST)
                avg_val_R = total_R / len(ASPECT_LIST)
                avg_val_F1 = total_F1 / len(ASPECT_LIST)
                
                logger.info(f"Epoch {epoch} [Macro-Avg] | P: {avg_val_P:.4f} | R: {avg_val_R:.4f} | F1: {avg_val_F1:.4f}")

                if avg_val_F1 > max_f1_score:
                    max_f1_score = avg_val_F1
                    logger.info(f"New Best F1-Score ({max_f1_score:.4f})! Saving model...")
                    save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth', model, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_best.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_best.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                    
                save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_last.pth', model, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_last.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_last.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_f1_score, scaler=scaler)
                print("\n")

    # --- 6. TEST (WITH BEAM SEARCH & FULL ROUGE) ---
    if args.do_eval and master_process:
        try:
            test_data = pd.read_json(f'{args.pretrained_data_dir}/test_with_iaog.json')
            test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
            test_loader = DataLoader(IAOGDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder), batch_size=args.eval_batch_size)
        except: return

        ckpt_path = f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth'
        # ckpt_path = f'/kaggle/input/iaog-bert-score/pytorch/30_epoch/4/seed_42_iaog_model_best.pth'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(model, DDP): model.module.load_state_dict(ckpt['model_state_dict'])
            else: model.load_state_dict(ckpt['model_state_dict'])
            
            # Load ResNet Image
            resimg_path = ckpt_path.replace("iaog_model", "resimg_model")
            if os.path.exists(resimg_path):
                resimg_ckpt = torch.load(resimg_path, map_location=device, weights_only=False)
                unwrap_resimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
                unwrap_resimg.load_state_dict(resimg_ckpt['model_state_dict'])
            
            # Load ResNet RoI
            resroi_path = ckpt_path.replace("iaog_model", "resroi_model")
            if os.path.exists(resroi_path):
                resroi_ckpt = torch.load(resroi_path, map_location=device, weights_only=False)
                unwrap_resroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
                unwrap_resroi.load_state_dict(resroi_ckpt['model_state_dict'])
                model.eval(); resnet_img.eval(); resnet_roi.eval()
                
        all_test_results = []
        
        # --- CHANGED: Storage for BERTScore (Test) ---
        test_preds = {asp: [] for asp in ASPECT_LIST}
        test_refs = {asp: [] for asp in ASPECT_LIST}
        
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
                        
                        test_preds[aspect_name].append(p)
                        test_refs[aspect_name].append(g)
                        
                        batch_results[i]["aspects"][aspect_name] = {"predict": p, "label": g}
                
                all_test_results.extend(batch_results)

        logger.info(f"Computing BERTScore for Test Set using model: {args.bert_score_model} ...")
        
        test_total_P, test_total_R, test_total_F1 = 0, 0, 0
        
        log_path = f"{args.output_dir}/iaog_test_predictions_formatted.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"TEST METRICS (BERTScore with {args.bert_score_model}):\n")
            f.write("-" * 50 + "\n")
            
            for asp in ASPECT_LIST:
                P, R, F1 = score(test_preds[asp], test_refs[asp], lang='vi', model_type=args.bert_score_model, verbose=False, device=device, num_layers=12)
                
                asp_P = P.mean().item()
                asp_R = R.mean().item()
                asp_F1 = F1.mean().item()
                
                test_total_P += asp_P
                test_total_R += asp_R
                test_total_F1 += asp_F1
                
                line = f"{asp:<15} | P: {asp_P:.4f} | R: {asp_R:.4f} | F1: {asp_F1:.4f}\n"
                logger.info(line.strip())
                f.write(line)
            
            avg_rP = test_total_P / len(ASPECT_LIST)
            avg_rR = test_total_R / len(ASPECT_LIST)
            avg_rF1 = test_total_F1 / len(ASPECT_LIST)

            f.write("-" * 50 + "\n")
            f.write(f"MACRO AVERAGE   | P: {avg_rP:.4f} | R: {avg_rR:.4f} | F1: {avg_rF1:.4f}\n")
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
        
        logger.info(f"***** TEST RESULTS (Macro Avg) *****")
        logger.info(f"Test Precision: {avg_rP:.4f}") 
        logger.info(f"Test Recall:    {avg_rR:.4f}") 
        logger.info(f"Test F1-Score:  {avg_rF1:.4f}") 
        logger.info(f"Formatted predictions saved to {log_path}")

if __name__ == '__main__':
    main()