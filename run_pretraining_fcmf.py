import torch
from text_preprocess import *
from iaog_dataset import IAOGDataset 
from fcmf_framework.fcmf_pretraining import FCMFSeq2Seq 
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

# --- HELPER FUNCTIONS ---

def save_model(path, model, optimizer, scheduler, epoch, best_score):
    """
    Save model checkpoint with all necessary states for resuming.
    Includes 'best_score' to track the metric (ROUGE-1) improvement.
    """
    if hasattr(model, 'module'): 
        model_state = model.module.state_dict()
    else: 
        model_state = model.state_dict()
        
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score, 
    }, path)

def main():
    parser = argparse.ArgumentParser()

    # --- PATH ARGUMENTS ---
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True,
                        help="Input data directory containing roi_data.csv")
    parser.add_argument("--pretrained_data_dir", default='../iaog-pretraining', type=str, required=True,
                        help="Directory containing pretraining json files (train_with_iaog.json)")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--pretrained_hf_model", default=None, type=str, required=True,
                        help="HuggingFace pretrained model name (e.g., uitnlp/visobert)")
    parser.add_argument('--image_dir', default='../vimacsa/image', help='Path to image folder')
    parser.add_argument('--resnet_label_path', default='/kaggle/input/resnet-output', help='Path to ResNet label jsons')
    
    # --- RESUME ---
    parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                        help="Path to checkpoint to resume training")

    # --- MODEL CONFIG ---
    parser.add_argument("--max_seq_length", default=170, type=int)
    parser.add_argument("--max_len_decoder", default=20, type=int)
    parser.add_argument("--num_imgs", default=7, type=int)
    parser.add_argument("--num_rois", default=4, type=int)
    parser.add_argument('--fine_tune_cnn', action='store_true', help="Whether to fine-tune ResNet")
    parser.add_argument("--alpha", default=0.8, type=float, help="Alpha for MDE")
    
    # --- TRAINING HYPERPARAMETERS ---
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
    
    # --- DISTRIBUTED TRAINING ---
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ddp", action='store_true')
    parser.add_argument("--list_aspect", nargs='+', default=[]) # Kept for compatibility

    args = parser.parse_args()

    # ==========================================================================================
    # 1. SETUP SYSTEM (Device, Logging, Seed)
    # ==========================================================================================
    if args.no_cuda:
        device = torch.device("cpu")
        ddp_local_rank = 0
        master_process = True
    elif args.ddp:
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(ddp_local_rank)
        dist.init_process_group(backend='nccl')
        device = f'cuda:{ddp_local_rank}'
        master_process = ddp_local_rank == 0
        ddp_world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ddp_local_rank = 0
        master_process = True
        ddp_world_size = 1

    # Setup Logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO if master_process else logging.WARN)
    logger = logging.getLogger(__name__)
    if master_process: os.makedirs(args.output_dir, exist_ok=True)

    # Adjust batch size for gradient accumulation
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ddp_world_size > 1: torch.cuda.manual_seed_all(args.seed)

    # ==========================================================================================
    # 2. PREPARE DATA & TOKENIZER
    # ==========================================================================================
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    # [CRITICAL] Add <iaog> token for decoder start
    tokenizer.add_special_tokens({'additional_special_tokens': ['<iaog>']})
    normalize_class = TextNormalize()

    # Load ResNet Metadata (Image/ROI Labels)
    try:
        roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv")
        roi_df['file_name'] = roi_df['file_name'] + '.png'
        
        # Determine json path (support Kaggle structure)
        json_path = args.resnet_label_path if os.path.exists(f'{args.resnet_label_path}/resnet152_image_label.json') else args.data_dir
        with open(f'{json_path}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
        with open(f'{json_path}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return

    # Load Datasets
    if args.do_train:
        train_data = pd.read_json(f'{args.pretrained_data_dir}/train_with_iaog.json')
        dev_data = pd.read_json(f'{args.pretrained_data_dir}/dev_with_iaog.json')
        
        # Preprocess text
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        
        # Split for DDP
        if ddp_world_size > 1:
            chunk = len(train_data) // ddp_world_size
            train_data = train_data.iloc[chunk*ddp_local_rank : chunk*(ddp_local_rank+1)]

        train_dataset = IAOGDataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder)
        dev_dataset = IAOGDataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder)

    # ==========================================================================================
    # 3. INITIALIZE MODEL & RESNETS
    # ==========================================================================================
    # Initialize FCMF Seq2Seq Model
    model = FCMFSeq2Seq(len(tokenizer), args.max_len_decoder, args.pretrained_hf_model, args.num_imgs, args.num_rois, args.alpha)
    # Resize embeddings for new tokens (<iaog>)
    model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))
    model.decoder.embedding = torch.nn.Embedding(len(tokenizer), model.decoder.num_hiddens) # Explicitly resize decoder embedding if not tied immediately or for safety
    
    # Initialize ResNets (Frozen or Fine-tuned)
    img_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(img_res, args.fine_tune_cnn, device).to(device)
    resnet_roi = myResNetRoI(roi_res, args.fine_tune_cnn, device).to(device)
    model = model.to(device)

    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        resnet_img = DDP(resnet_img, device_ids=[ddp_local_rank])
        resnet_roi = DDP(resnet_roi, device_ids=[ddp_local_rank])

    # ==========================================================================================
    # 4. OPTIMIZER & SCHEDULER
    # ==========================================================================================
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100) # Ignore padded labels
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    num_train_steps = len(train_dataset) // args.train_batch_size * args.num_train_epochs if args.do_train else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps*args.warmup_proportion, num_training_steps=num_train_steps)

    # ==========================================================================================
    # 5. CHECKPOINT RESUMING
    # ==========================================================================================
    start_epoch = 0
    max_rouge1 = 0.0 # [UPDATED] Track Best ROUGE-1 instead of Loss

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        
        # Restore best score if available
        if 'best_score' in ckpt:
            max_rouge1 = ckpt['best_score']
            
        logger.info(f"Resumed from epoch {start_epoch}, Current Best ROUGE-1: {max_rouge1}")

    # ==========================================================================================
    # 6. TRAINING LOOP
    # ==========================================================================================
    # Initialize Scorer for Validation/Test
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    if args.do_train:
        train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset) if args.ddp else RandomSampler(train_dataset), batch_size=args.train_batch_size)
        dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            if args.ddp: train_loader.sampler.set_epoch(epoch)
            
            # --- TRAIN PHASE ---
            model.train(); resnet_img.train(); resnet_roi.train()
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, disable=not master_process)
            tr_loss = 0
            
            for step, batch in enumerate(pbar):
                pbar.set_description(f"Epoch {epoch}")
                
                # Move to device
                batch = tuple(t.to(device) for t in batch)
                (t_img_f, roi_img_f, roi_coors, all_labels, all_dec_ids, all_dec_mask, 
                 all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _) = batch
                
                roi_img_f = roi_img_f.float()

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    # 1. Feature Extraction (Once per batch)
                    with torch.no_grad():
                        enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        roi_embeds = torch.stack(enc_rois, dim=1)

                    # 2. Forward Loop (6 Aspects)
                    total_loss = 0
                    for id_asp in range(6): # Loop over 6 aspects
                        # Slice data for current aspect
                        logits = model(enc_X=all_enc_ids[:,id_asp], dec_X=all_dec_ids[:,id_asp], 
                                       visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                       token_type_ids=all_enc_type[:,id_asp], attention_mask=all_enc_mask[:,id_asp], 
                                       added_attention_mask=all_add_mask[:,id_asp], is_train=True)
                        
                        # [FIX] Use reshape instead of view to handle non-contiguous tensors
                        loss_asp = criterion(logits.reshape(-1, logits.size(-1)), all_labels[:,id_asp].reshape(-1))
                        total_loss += loss_asp

                    # Gradient Accumulation
                    if args.gradient_accumulation_steps > 1: 
                        total_loss /= args.gradient_accumulation_steps

                # 3. Backward
                if args.fp16: scaler.scale(total_loss).backward()
                else: total_loss.backward()
                
                tr_loss += total_loss.item()

                # 4. Optimization Step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.fp16: scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    scheduler.step(); optimizer.zero_grad()
                    
                    pbar.set_postfix(loss=total_loss.item() * args.gradient_accumulation_steps)

            # --- VALIDATION PHASE (Calculate ROUGE) ---
            if master_process and args.do_eval:
                model.eval(); resnet_img.eval(); resnet_roi.eval()
                val_loss = 0
                val_rouge1_scores = []
                val_rougeL_scores = []
                
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Eval", leave=False):
                        batch = tuple(t.to(device) for t in batch)
                        (t_img_f, roi_img_f, roi_coors, all_labels, all_dec_ids, _, 
                         all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _) = batch
                        
                        # Feature Extraction
                        enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                        roi_embeds = torch.stack(enc_rois, dim=1)

                        for id_asp in range(6):
                            # A. Calculate Validation Loss
                            logits = model(enc_X=all_enc_ids[:,id_asp], dec_X=all_dec_ids[:,id_asp],
                                           visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                           token_type_ids=all_enc_type[:,id_asp], attention_mask=all_enc_mask[:,id_asp], 
                                           added_attention_mask=all_add_mask[:,id_asp], is_train=True)
                            
                            val_loss += criterion(logits.reshape(-1, logits.size(-1)), all_labels[:,id_asp].reshape(-1)).item()

                            # B. Calculate ROUGE (Greedy Generation)
                            preds = torch.argmax(logits, dim=-1)
                            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                            
                            lbls = all_labels[:,id_asp].cpu().numpy()
                            # Clean labels (remove -100 masking)
                            lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
                            decoded_lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
                            
                            for p, g in zip(decoded_preds, decoded_lbls):
                                s = scorer.score(g, p)
                                val_rouge1_scores.append(s['rouge1'].fmeasure)
                                val_rougeL_scores.append(s['rougeL'].fmeasure)

                # Calculate Average Metrics
                avg_val_loss = val_loss / (len(dev_loader) * 6)
                avg_rouge1 = np.mean(val_rouge1_scores) if val_rouge1_scores else 0.0
                avg_rougeL = np.mean(val_rougeL_scores) if val_rougeL_scores else 0.0

                logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | ROUGE-1: {avg_rouge1:.4f} | ROUGE-L: {avg_rougeL:.4f}")

                # [UPDATED] Save BEST model based on ROUGE-1
                if avg_rouge1 > max_rouge1:
                    max_rouge1 = avg_rouge1
                    logger.info(f"New Best ROUGE-1 ({max_rouge1:.4f})! Saving model...")
                    
                    save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth', model, optimizer, scheduler, epoch, best_score=max_rouge1)
                    # Also save ResNets
                    save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_best.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_rouge1)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_best.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_rouge1)
                
                # Save LATEST model (for resuming)
                save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_last.pth', model, optimizer, scheduler, epoch, best_score=max_rouge1)
                save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_last.pth', resnet_img, optimizer, scheduler, epoch, best_score=max_rouge1)
                save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_last.pth', resnet_roi, optimizer, scheduler, epoch, best_score=max_rouge1)

    # ==========================================================================================
    # 7. TEST PHASE (With detailed Logging)
    # ==========================================================================================
    if args.do_eval and master_process:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        try:
            test_data = pd.read_json(f'{args.pretrained_data_dir}/test_with_iaog.json')
            test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
            test_loader = DataLoader(IAOGDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois, args.max_len_decoder), batch_size=args.eval_batch_size)
        except Exception as e: 
            logger.error(f"Cannot load test set: {e}")
            return

        # Load BEST Checkpoint
        # ckpt_path = f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth'
        ckpt_path = f'/kaggle/input/iaog-best-6-aspect/pytorch/16_epoch/1/seed_42_iaog_model_best_6_aspect_loss.pth'
        if os.path.exists(ckpt_path):
            logger.info(f"Loading Best Checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(model, DDP): model.module.load_state_dict(ckpt['model_state_dict'])
            else: model.load_state_dict(ckpt['model_state_dict'])
        else:
            logger.warning("No best checkpoint found! Testing with current model.")
        
        model.eval(); resnet_img.eval(); resnet_roi.eval()
        
        all_preds = []
        all_labels = []
        all_rouge1 = []
        all_rougeL = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                batch = tuple(t.to(device) for t in batch)
                (t_img_f, roi_img_f, roi_coors, all_dec_lbls, all_dec_ids, _, 
                 all_enc_ids, all_enc_type, all_enc_mask, all_add_mask, _) = batch

                # Feature Extract
                enc_imgs = [resnet_img(t_img_f[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                vis_embeds = torch.stack(enc_imgs, dim=1)
                enc_rois = [torch.stack([resnet_roi(roi_img_f[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                roi_embeds = torch.stack(enc_rois, dim=1)

                # Loop Aspects
                for id_asp in range(6):
                    logits = model(enc_X=all_enc_ids[:,id_asp], dec_X=all_dec_ids[:,id_asp],
                                   visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                   token_type_ids=all_enc_type[:,id_asp], attention_mask=all_enc_mask[:,id_asp], 
                                   added_attention_mask=all_add_mask[:,id_asp], is_train=True)
                    
                    # Generation
                    preds = torch.argmax(logits, dim=-1)
                    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                    
                    # Labels
                    lbls = all_dec_lbls[:,id_asp].cpu().numpy()
                    lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
                    decoded_lbls = tokenizer.batch_decode(lbls, skip_special_tokens=True)
                    
                    # Accumulate for logs
                    for p, g in zip(decoded_preds, decoded_lbls):
                        all_preds.append(p)
                        all_labels.append(g)
                        s = scorer.score(g, p)
                        all_rouge1.append(s['rouge1'].fmeasure)
                        all_rougeL.append(s['rougeL'].fmeasure)

        # Log Statistics
        avg_r1 = np.mean(all_rouge1)
        avg_rL = np.mean(all_rougeL)
        
        logger.info(f"***** TEST RESULTS *****")
        logger.info(f"Test ROUGE-1: {avg_r1:.4f}")
        logger.info(f"Test ROUGE-L: {avg_rL:.4f}")

        # [UPDATED] Save Predictions Log
        log_path = f"{args.output_dir}/test_predictions_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"TEST METRICS:\n")
            f.write(f"ROUGE-1: {avg_r1:.4f}\n")
            f.write(f"ROUGE-L: {avg_rL:.4f}\n")
            f.write("="*50 + "\n")
            f.write("PREDICTION vs LABEL\n")
            f.write("="*50 + "\n")
            for i in range(len(all_preds)):
                f.write(f"Sample {i}:\n")
                f.write(f"  Pred:  {all_preds[i]}\n")
                f.write(f"  Label: {all_labels[i]}\n")
                f.write("-" * 20 + "\n")
        
        logger.info(f"Predictions saved to {log_path}")

if __name__ == '__main__':
    main()