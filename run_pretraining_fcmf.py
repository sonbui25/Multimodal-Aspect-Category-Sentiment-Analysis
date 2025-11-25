import torch
from text_preprocess import *
from iaog_dataset import IAOGDataset 
from fcmf_framework.fcmf_pretraining import FCMFSeq2Seq 
from sklearn.metrics import precision_recall_fscore_support
import argparse
import logging
import random
from tqdm.auto import tqdm, trange
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from underthesea import word_tokenize, text_normalize
from fcmf_framework.resnet_utils import *
from torchvision.models import resnet152, ResNet152_Weights, resnet50, ResNet50_Weights
from fcmf_framework.optimization import BertAdam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json
from torch.cuda.amp import autocast
import os
import glob
from rouge_score import rouge_scorer

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def save_model(path, model, optimizer, scheduler, epoch):
    """
    Lưu checkpoint đầy đủ trạng thái để có thể Resume sau này.
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
    }, path)

def load_model(path):
    check_point = torch.load(path, map_location='cpu')
    return check_point

def main():
    parser = argparse.ArgumentParser()

    # --- INPUT/OUTPUT PATHS ---
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True,
                        help="The input data dir. Should contain train/dev/test json files.")
    parser.add_argument("--pretrained_data_dir", default='../iaog-pretraining', type=str, required=True,
                        help="The input data dir. Should contain train/dev/test json files.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("-- ", default=None, type=str, required=True,
                        help="Pretrained huggingface model (e.g. xlm-roberta-base).") ##Note pretrained_hf_model is path for model like xlm-roberta-base, not path for iaog pretraining weights
    parser.add_argument('--image_dir', default='../vimacsa/image', help='path to images')
    parser.add_argument('--resnet_label_path', default='/kaggle/input/resnet-output', help='Directory containing resnet label jsons')
    
    # --- RESUME TRAINING ---
    parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                        help="Path to the checkpoint .pth file to resume training from.")

    # --- MODEL CONFIG ---
    parser.add_argument("--max_seq_length", default=170, type=int)
    parser.add_argument("--max_len_decoder", default=20, type=int)
    parser.add_argument("--num_imgs", default=7, type=int)
    parser.add_argument("--num_rois", default=4, type=int)
    parser.add_argument('--fine_tune_cnn', action='store_true')

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
    
    # --- SYSTEM ---
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ddp", action='store_true')

    # Ignored args but kept for compatibility
    parser.add_argument("--list_aspect", nargs='+', default=[]) 

    args = parser.parse_args()

    # ==========================================================================================
    # 1. SETUP DEVICE & LOGGER
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

    print(f"Running on device: {device}")

    if master_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(f"{args.output_dir}/training_iaog.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Arguments: {args}")

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    # Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ddp_world_size > 1:
        torch.cuda.manual_seed_all(args.seed)

    # ==========================================================================================
    # 2. TOKENIZER & METADATA
    # ==========================================================================================
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    # [CRITICAL] Add special token <iaog>
    tokenizer.add_special_tokens({'additional_special_tokens': ['<iaog>']})
    
    normalize_class = TextNormalize()

    # Load ResNet Metadata
    try:
        roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv")
        roi_df['file_name'] = roi_df['file_name'] + '.png'
        
        json_path = args.resnet_label_path if os.path.exists(f'{args.resnet_label_path}/resnet152_image_label.json') else args.data_dir
        with open(f'{json_path}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
        with open(f'{json_path}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)
    except Exception as e:
        if master_process: logger.error(f"Error loading metadata: {e}")
        return

    # ==========================================================================================
    # 3. PREPARE DATASETS
    # ==========================================================================================
    if args.do_train:
        train_data = pd.read_json(f'{args.pretrained_data_dir}/train_with_iaog.json')
        dev_data = pd.read_json(f'{args.pretrained_data_dir}/dev_with_iaog.json')
        
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))

        if ddp_world_size > 1:
            chunk_size = len(train_data) // ddp_world_size
            train_data = train_data.iloc[chunk_size*ddp_local_rank : chunk_size*(ddp_local_rank+1)]

        train_dataset = IAOGDataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, 
                                    args.num_imgs, args.num_rois, args.max_len_decoder)
        dev_dataset = IAOGDataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, 
                                  args.num_imgs, args.num_rois, args.max_len_decoder)

    # ==========================================================================================
    # 4. INITIALIZE MODEL
    # ==========================================================================================
    model = FCMFSeq2Seq(args.pretrained_hf_model, vocab_size=len(tokenizer), max_len_decoder=args.max_len_decoder)
    model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))
    model.decoder.embedding = torch.nn.Embedding(len(tokenizer), model.decoder.num_hiddens)

    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(resnet=img_res_model, if_fine_tune=args.fine_tune_cnn, device=device)
    resnet_roi = myResNetRoI(resnet=roi_res_model, if_fine_tune=args.fine_tune_cnn, device=device)

    model = model.to(device)

    # DDP Setup
    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        resnet_img = DDP(resnet_img, device_ids=[ddp_local_rank])
        resnet_roi = DDP(resnet_roi, device_ids=[ddp_local_rank])
    elif torch.cuda.device_count() > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)
        resnet_img = torch.nn.DataParallel(resnet_img)
        resnet_roi = torch.nn.DataParallel(resnet_roi)

    # ==========================================================================================
    # 5. OPTIMIZER & SCHEDULER
    # ==========================================================================================
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Scheduler Setup
    num_train_steps = 0
    if args.do_train:
        num_train_steps = len(train_dataset) // args.train_batch_size * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps*args.warmup_proportion, num_training_steps=num_train_steps)

    # ==========================================================================================
    # 6. RESUME CHECKPOINT LOGIC
    # ==========================================================================================
    start_epoch = 0
    min_val_loss = float('inf')

    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if os.path.isfile(ckpt_path):
            if master_process: logger.info(f"Loading checkpoint from {ckpt_path}")
            
            # Load IAOG Model
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(model, (DDP, torch.nn.DataParallel)):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load Optimizer/Scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            # Load ResNets (Assumes naming convention: seed_42_iaog_model_best.pth -> seed_42_resimg_model_best.pth)
            resimg_path = ckpt_path.replace("iaog_model", "resimg_model")
            resroi_path = ckpt_path.replace("iaog_model", "resroi_model")
            
            if os.path.exists(resimg_path):
                resimg_ckpt = torch.load(resimg_path, map_location=device)
                unwrap_resimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
                unwrap_resimg.load_state_dict(resimg_ckpt['model_state_dict'])
            
            if os.path.exists(resroi_path):
                resroi_ckpt = torch.load(resroi_path, map_location=device)
                unwrap_resroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
                unwrap_resroi.load_state_dict(resroi_ckpt['model_state_dict'])

            if master_process: logger.info(f"Resumed from Epoch {start_epoch}")
        else:
            if master_process: logger.warning(f"Checkpoint {ckpt_path} not found. Starting from scratch.")

    # ==========================================================================================
    # 7. TRAINING LOOP
    # ==========================================================================================
    if args.do_train:
        sampler = DistributedSampler(train_dataset) if args.ddp else RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=args.eval_batch_size)

        for epoch in range(start_epoch, int(args.num_train_epochs)):
            if args.ddp: sampler.set_epoch(epoch)
            
            if master_process: logger.info(f"***** Epoch {epoch} *****")
            
            model.train()
            resnet_img.train()
            resnet_roi.train()
            optimizer.zero_grad()
            
            train_loss = 0
            
            # --- Train Step ---
            pbar = tqdm(train_dataloader, disable=not master_process)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                (t_img_feat, roi_img_feat, roi_coors, labels, dec_input_ids, _, 
                 enc_input_ids, enc_type_ids, enc_mask, added_mask, valid_lens) = batch

                with autocast(enabled=args.fp16):
                    # Feature Extract
                    enc_imgs = []
                    for i in range(args.num_imgs):
                        feat = resnet_img(t_img_feat[:, i]).view(-1, 2048, 49).permute(0, 2, 1)
                        enc_imgs.append(feat)
                    vis_embeds = torch.stack(enc_imgs, dim=1)

                    enc_rois = []
                    for i in range(args.num_imgs):
                        roi_list = [resnet_roi(roi_img_feat[:, i, r]).squeeze(1) for r in range(args.num_rois)]
                        enc_rois.append(torch.stack(roi_list, dim=1))
                    roi_embeds = torch.stack(enc_rois, dim=1)

                    # Forward
                    logits = model(
                        enc_X=enc_input_ids,
                        dec_X=dec_input_ids,
                        visual_embeds_att=vis_embeds,
                        roi_embeds_att=roi_embeds,
                        roi_coors=roi_coors,
                        token_type_ids=enc_type_ids,
                        attention_mask=enc_mask,
                        added_attention_mask=added_mask,
                        source_valid_len=valid_lens,
                        is_train=True
                    )
                    
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                # Backward
                scaler.scale(loss).backward() if args.fp16 else loss.backward()
                train_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer) if args.fp16 else optimizer.step()
                    if args.fp16: scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.set_postfix(loss=loss.item())

            # --- Eval Step (After each Epoch) ---
            if master_process and args.do_eval:
                model.eval()
                resnet_img.eval()
                resnet_roi.eval()
                
                total_val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(dev_dataloader, desc="Evaluating Dev", leave=False):
                        batch = tuple(t.to(device) for t in batch)
                        (t_img_feat, roi_img_feat, roi_coors, labels, dec_input_ids, _, 
                         enc_input_ids, enc_type_ids, enc_mask, added_mask, valid_lens) = batch
                        
                        # Feature Extract
                        enc_imgs = [resnet_img(t_img_feat[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                        vis_embeds = torch.stack(enc_imgs, dim=1)
                        
                        enc_rois = []
                        for i in range(args.num_imgs):
                            roi_list = [resnet_roi(roi_img_feat[:,i,r]).squeeze(1) for r in range(args.num_rois)]
                            enc_rois.append(torch.stack(roi_list, dim=1))
                        roi_embeds = torch.stack(enc_rois, dim=1)

                        logits = model(enc_X=enc_input_ids, dec_X=dec_input_ids,
                                       visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                                       token_type_ids=enc_type_ids, attention_mask=enc_mask,
                                       added_attention_mask=added_mask, source_valid_len=valid_lens, is_train=True)
                        
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(dev_dataloader)
                logger.info(f"Dev Loss Epoch {epoch}: {avg_val_loss}")

                # Save BEST Checkpoint (Only Master)
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    logger.info("New Best Model! Saving...")
                    save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth', model, optimizer, scheduler, epoch)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_best.pth', resnet_img, optimizer, scheduler, epoch)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_best.pth', resnet_roi, optimizer, scheduler, epoch)

                # Save LATEST Checkpoint (For Resuming)
                save_model(f'{args.output_dir}/seed_{args.seed}_iaog_model_last.pth', model, optimizer, scheduler, epoch)
                save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_last.pth', resnet_img, optimizer, scheduler, epoch)
                save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_last.pth', resnet_roi, optimizer, scheduler, epoch)

    # ==========================================================================================
    # 8. TEST EVALUATION (FULL) - ROUGE SCORE
    # ==========================================================================================
    if args.do_eval and master_process:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        
        # 8.1 Load Test Data
        try:
            test_data = pd.read_json(f'{args.pretrained_data_dir}/test_with_iaog.json')
            test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
            test_dataset = IAOGDataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, 
                                       args.num_imgs, args.num_rois, args.max_len_decoder)
            test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)
        except Exception as e:
            logger.error(f"Could not load test data: {e}")
            return

        # 8.2 Load BEST Model
        best_path = f'{args.output_dir}/seed_{args.seed}_iaog_model_best.pth'
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            
            # Handle DDP wrapping
            if isinstance(model, (DDP, torch.nn.DataParallel)):
                model.module.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt['model_state_dict'])
                
            # Load ResNets
            rimg_ckpt = torch.load(best_path.replace("iaog", "resimg"), map_location=device)
            unwrap_rimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
            unwrap_rimg.load_state_dict(rimg_ckpt['model_state_dict'])
            
            rroi_ckpt = torch.load(best_path.replace("iaog", "resroi"), map_location=device)
            unwrap_rroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
            unwrap_rroi.load_state_dict(rroi_ckpt['model_state_dict'])
        else:
            logger.warning("No best checkpoint found in output_dir. Testing with current model weights.")

        # 8.3 Run Test Loop
        model.eval()
        resnet_img.eval()
        resnet_roi.eval()
        total_test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating Test"):
                batch = tuple(t.to(device) for t in batch)
                (t_img_feat, roi_img_feat, roi_coors, labels, dec_input_ids, _, 
                 enc_input_ids, enc_type_ids, enc_mask, added_mask, valid_lens) = batch

                # Feature Extract
                enc_imgs = [resnet_img(t_img_feat[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)]
                vis_embeds = torch.stack(enc_imgs, dim=1)
                enc_rois = [torch.stack([resnet_roi(roi_img_feat[:,i,r]).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)]
                roi_embeds = torch.stack(enc_rois, dim=1)

                # Forward
                logits = model(enc_X=enc_input_ids, dec_X=dec_input_ids,
                               visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                               token_type_ids=enc_type_ids, attention_mask=enc_mask,
                               added_attention_mask=added_mask, source_valid_len=valid_lens, is_train=True)
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_test_loss += loss.item()

                # --- GENERATION FOR METRICS (Greedy Selection) ---
                preds_batch = torch.argmax(logits, dim=-1)
                decoded_preds = tokenizer.batch_decode(preds_batch, skip_special_tokens=True)
                
                # Clean labels
                labels_cpu = labels.cpu().numpy()
                labels_cpu = np.where(labels_cpu != -100, labels_cpu, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels_cpu, skip_special_tokens=True)

                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)

        avg_test_loss = total_test_loss / len(test_dataloader)
        
        # --- [NEW] ROUGE SCORE CALCULATION ---
        logger.info("Calculating ROUGE Score...")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        r1, r2, rl = [], [], []
        for pred, label in zip(all_preds, all_labels):
            scores = scorer.score(label, pred) # score(target, prediction)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)
            
        avg_r1 = np.mean(r1)
        avg_r2 = np.mean(r2)
        avg_rl = np.mean(rl)

        logger.info(f"***** TEST RESULTS *****")
        logger.info(f"Test Loss: {avg_test_loss}")
        logger.info(f"ROUGE-1: {avg_r1}")
        logger.info(f"ROUGE-2: {avg_r2}")
        logger.info(f"ROUGE-L: {avg_rl}")
        
        with open(f"{args.output_dir}/test_results_iaog.txt", "w") as f:
            f.write(f"Test Loss: {avg_test_loss}\n")
            f.write(f"ROUGE-1: {avg_r1}\n")
            f.write(f"ROUGE-2: {avg_r2}\n")
            f.write(f"ROUGE-L: {avg_rl}\n")
            
            f.write("\n--- Sample Predictions ---\n")
            for i in range(min(5, len(all_preds))):
                f.write(f"Pred: {all_preds[i]}\nRef : {all_labels[i]}\n\n")

if __name__ == '__main__':
    main()