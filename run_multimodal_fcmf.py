import torch
import torch.nn as nn
from text_preprocess import *
from vimacsa_dataset import *
from fcmf_framework.fcmf_multimodal import FCMF
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

# Map numeric labels back to string for logging
POLARITY_MAP = {0: 'None', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ \
      = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
    return p_macro, r_macro, f_macro

def save_model(path, model, optimizer, scheduler, epoch, best_score=0.0):
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)

def load_model(path):
    check_point = torch.load(path, map_location='cpu')
    return check_point

def main():
    parser = argparse.ArgumentParser()

    # --- PATH ARGUMENTS ---
    parser.add_argument("--data_dir", default='../vimacsa', type=str, required=True,
                        help="The input data dir containing .json files.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--image_dir', default='../vimacsa/image', help='path to images')
    parser.add_argument('--resnet_label_path', default='/kaggle/input/resnet-output', help='Resnet labels path')

    # --- MODEL ARGUMENTS ---
    parser.add_argument("--pretrained_hf_model", default=None, type=str, required=True,
                        help="Pre-trained huggingface model and tokenizer (e.g. xlm-roberta-base).")
    
    # [NEW] Argument to load the Encoder weights from the IAOG Pretraining phase
    parser.add_argument("--pretrained_iaog_path", default=None, type=str,
                        help="Path to the IAOG best checkpoint to initialize Encoder weights from.")
    
    # [NEW] Argument to resume training from a specific checkpoint
    parser.add_argument("--resume_from_checkpoint", default=None, type=str,
                        help="Path to the checkpoint .pth file to resume training from.")

    # --- HYPERPARAMETERS ---
    parser.add_argument("--list_aspect", default=['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area'],
                        nargs='+', help = "List of predefined Aspect.")
    parser.add_argument("--num_polarity", default=4, type=int, help="Number of sentiment polarity.")
    parser.add_argument("--num_imgs", default=7, type=int, help="Number of images.")
    parser.add_argument("--num_rois", default=7, type=int, help="Number of RoIs.")
    parser.add_argument("--max_seq_length", default=170, type=int)
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float, help="Total number of training epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup.")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision")
    parser.add_argument('--alpha', type=float, default=0.7, help="Alpha value for keeping strong visual features")
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    
    # --- SYSTEM ARGUMENTS ---
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--ddp", action='store_true', help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    # ==========================================================================================
    # 1. SETUP DEVICE & LOGGING
    # ==========================================================================================
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    elif args.ddp:
        assert torch.cuda.is_available()
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    elif not args.ddp:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running on device:{ddp_local_rank}")
    
    if master_process:
        print("===================== RUN Fine-grained Cross-modal Fusion (MACSA) =====================")
        os.makedirs(args.output_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(f'{args.output_dir}/training_fcmf.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, ddp_world_size, bool(args.ddp), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ddp_world_size > 1:
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.init_process_group(backend='nccl')

    # ==========================================================================================
    # 2. LOAD TOKENIZER & METADATA
    # ==========================================================================================
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
        special_tokens = {'additional_special_tokens': ['<iaog>']}
        tokenizer.add_special_tokens(special_tokens)
    except:
        raise ValueError("Wrong pretrained model.")

    normalize_class = TextNormalize()
    ASPECT = args.list_aspect

    try:
        roi_df = pd.read_csv(f"{args.data_dir}/roi_data.csv")
        roi_df['file_name'] = roi_df['file_name'] + '.png'
        
        json_path = args.resnet_label_path if os.path.exists(f'{args.resnet_label_path}/resnet152_image_label.json') else args.data_dir
        with open(f'{json_path}/resnet152_image_label.json') as imf: dict_image_aspect = json.load(imf)
        with open(f'{json_path}/resnet152_roi_label.json') as rf: dict_roi_aspect = json.load(rf)
    except Exception as e:
        raise ValueError(f"Error loading metadata: {e}")

    if args.do_train:
        train_data = pd.read_json(f'{args.data_dir}/train.json')
        dev_data = pd.read_json(f'{args.data_dir}/dev.json')
        
        train_data['comment'] = train_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        dev_data['comment'] = dev_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))

        if ddp_world_size > 1:
            num_splitted_train = train_data.shape[0] // ddp_world_size
            train_data = train_data.iloc[num_splitted_train*ddp_local_rank: num_splitted_train*(ddp_local_rank + 1),:]

        train_dataset = MACSADataset(train_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        dev_dataset = MACSADataset(dev_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)

    # ==========================================================================================
    # 3. MODEL INITIALIZATION
    # ==========================================================================================
    model = FCMF(pretrained_path=args.pretrained_hf_model,
                 num_labels=args.num_polarity,
                 num_imgs=args.num_imgs,
                 num_roi=args.num_rois,
                 alpha=args.alpha)
    
    # [IMPORTANT] Resize embedding to match tokenizer
    model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))

    img_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(resnet=img_res_model, if_fine_tune=args.fine_tune_cnn, device=device)
    resnet_roi = myResNetRoI(resnet=roi_res_model, if_fine_tune=args.fine_tune_cnn, device=device)

    model = model.to(device)

    if args.ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        resnet_img = DDP(resnet_img, device_ids=[ddp_local_rank])
        resnet_roi = DDP(resnet_roi, device_ids=[ddp_local_rank])
    elif torch.cuda.device_count() > 1 and not args.ddp:
        model = torch.nn.DataParallel(model)
        resnet_img = torch.nn.DataParallel(resnet_img)
        resnet_roi = torch.nn.DataParallel(resnet_roi)

    # ==========================================================================================
    # 4. OPTIMIZER & SCHEDULER
    # ==========================================================================================
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    encoder_params = []
    head_params = []
    head_names = ['classifier', 'text_pooler'] 
    
    for n, p in model.named_parameters():
        if any(nd in n for nd in head_names):
            head_params.append((n, p))
        else:
            encoder_params.append((n, p))

    optimizer_grouped_parameters = [
        # Encoder Group: Low LR
        {
            'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': 7e-5 
        },
        {
            'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': 7e-5
        },
        # Head Group: Higher LR
        {
            'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
            'lr': 7e-4 
        },
        {
            'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': 7e-4
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.fp16:
        scaler = torch.amp.GradScaler('cuda') 

    num_train_steps = 0
    if args.do_train:
        num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_train_steps*args.warmup_proportion, num_training_steps=num_train_steps)

    # ==========================================================================================
    # 5. CHECKPOINT LOADING
    # ==========================================================================================
    start_epoch = 0
    max_f1 = 0.0

    # A. RESUME
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if os.path.isfile(checkpoint_path):
            if master_process: logger.info(f"--> Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if isinstance(model, (DDP, torch.nn.DataParallel)):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            resimg_path = checkpoint_path.replace("fcmf_model", "resimg_model")
            resroi_path = checkpoint_path.replace("fcmf_model", "resroi_model")
            
            if os.path.exists(resimg_path):
                resimg_ckpt = torch.load(resimg_path, map_location=device)
                unwrap_resimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
                unwrap_resimg.load_state_dict(resimg_ckpt['model_state_dict'])
                
            if os.path.exists(resroi_path):
                resroi_ckpt = torch.load(resroi_path, map_location=device)
                unwrap_resroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
                unwrap_resroi.load_state_dict(resroi_ckpt['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            max_f1 = checkpoint.get('best_score', 0.0)
            if master_process: logger.info(f"--> Resumed successfully from Epoch {start_epoch} with Max F1: {max_f1}")
        else:
            if master_process: logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    # B. LOAD PRETRAINED ENCODER
    elif args.pretrained_iaog_path and os.path.isfile(args.pretrained_iaog_path):
        if master_process: logger.info(f"--> Loading Encoder weights from Pretraining: {args.pretrained_iaog_path}")
        iaog_ckpt = torch.load(args.pretrained_iaog_path, map_location='cpu', weights_only=False)
        iaog_state_dict = iaog_ckpt['model_state_dict']
        encoder_state_dict = {k: v for k, v in iaog_state_dict.items() if k.startswith('encoder.')}
        model_to_load = model.module if hasattr(model, 'module') else model 
        missing_keys, unexpected_keys = model_to_load.load_state_dict(encoder_state_dict, strict=False) 
        
        if master_process: 
            logger.info(f"--> Pretrained Encoder loaded successfully.")
            logger.info(f"    Keys loaded: {len(encoder_state_dict)}")
            logger.info(f"    Missing keys (expected): {len(missing_keys)}")
    else:
        if master_process: logger.info("--> No checkpoint or pretrained IAOG path provided. Training from scratch.")

    # ==========================================================================================
    # 6. TRAINING LOOP
    # ==========================================================================================
    if args.do_train:
        if not args.ddp:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        dev_sampler = SequentialSampler(dev_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size)

        for train_idx in range(start_epoch, int(args.num_train_epochs)):
            if args.ddp: train_sampler.set_epoch(train_idx)
            if master_process: logger.info(f"********** Epoch: {train_idx} **********")
            
            model.train(); resnet_img.train(); resnet_roi.train()
            optimizer.zero_grad()

            with tqdm(train_dataloader, position=0, leave=True, disable=not master_process, dynamic_ncols=True) as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {train_idx}")
                    
                    # Unpack batch (text is last element, use _ to ignore)
                    batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                    (t_img_features, roi_img_features, roi_coors, 
                    all_input_ids, all_token_types_ids, all_attn_mask, 
                    all_added_input_mask, all_label_id, _) = batch

                    # [CRITICAL] Fix DoubleTensor error by casting to float
                    roi_img_features = roi_img_features.float()

                    with torch.amp.autocast('cuda', enabled=args.fp16):
                        # Feature Extraction
                        encoded_img = []
                        for img_idx in range(args.num_imgs):
                            img_f = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                            encoded_img.append(img_f)
                        
                        encoded_roi = []
                        for img_idx in range(args.num_imgs):
                            roi_list = [resnet_roi(roi_img_features[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                            encoded_roi.append(torch.stack(roi_list, dim=1))
                        
                        vis_embeds = torch.stack(encoded_img, dim=1)
                        roi_embeds = torch.stack(encoded_roi, dim=1)

                        # Loop Aspects
                        all_asp_loss = 0
                        for id_asp in range(len(ASPECT)):
                            logits = model(
                                input_ids=all_input_ids[:,id_asp,:],
                                token_type_ids=all_token_types_ids[:,id_asp,:],
                                attention_mask=all_attn_mask[:,id_asp,:],
                                added_attention_mask=all_added_input_mask[:,id_asp,:],
                                visual_embeds_att=vis_embeds,
                                roi_embeds_att=roi_embeds,
                                roi_coors=roi_coors
                            )
                            loss = criterion(logits, all_label_id[:,id_asp])
                            all_asp_loss += loss

                        if args.gradient_accumulation_steps > 1:
                            all_asp_loss = all_asp_loss / args.gradient_accumulation_steps

                        if args.fp16: scaler.scale(all_asp_loss).backward()
                        else: all_asp_loss.backward()

                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if args.fp16: scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            if args.fp16: scaler.step(optimizer); scaler.update()
                            else: optimizer.step()
                            scheduler.step(); optimizer.zero_grad()
                            tepoch.set_postfix(loss=all_asp_loss.item() * args.gradient_accumulation_steps)

            # --- Evaluation ---
            if master_process and args.do_eval:
                logger.info("***** Running evaluation on Dev Set *****")
                model.eval(); resnet_img.eval(); resnet_roi.eval()
                true_label_list = {asp:[] for asp in ASPECT}
                pred_label_list = {asp:[] for asp in ASPECT}
                idx2asp = {i:v for i,v in enumerate(ASPECT)}

                with torch.no_grad():
                    for batch in tqdm(dev_dataloader, desc="Evaluating Dev", leave=False):
                        batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                        t_img_features, roi_img_features, roi_coors, \
                        all_input_ids, all_token_types_ids, all_attn_mask, \
                        all_added_input_mask, all_label_id, _ = batch

                        # [CRITICAL] Fix DoubleTensor error in Eval
                        roi_img_features = roi_img_features.float()

                        encoded_img = []
                        for img_idx in range(args.num_imgs):
                            img_f = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                            encoded_img.append(img_f)
                        
                        encoded_roi = []
                        for img_idx in range(args.num_imgs):
                            roi_list = [resnet_roi(roi_img_features[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                            encoded_roi.append(torch.stack(roi_list, dim=1))
                        
                        vis_embeds = torch.stack(encoded_img, dim=1)
                        roi_embeds = torch.stack(encoded_roi, dim=1)

                        for id_asp in range(len(ASPECT)):
                            logits = model(
                                input_ids=all_input_ids[:,id_asp,:],
                                token_type_ids=all_token_types_ids[:,id_asp,:],
                                attention_mask=all_attn_mask[:,id_asp,:],
                                added_attention_mask=all_added_input_mask[:,id_asp,:],
                                visual_embeds_att=vis_embeds,
                                roi_embeds_att=roi_embeds,
                                roi_coors=roi_coors
                            )
                            pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                            label = all_label_id[:,id_asp].cpu().numpy()
                            true_label_list[idx2asp[id_asp]].append(label)
                            pred_label_list[idx2asp[id_asp]].append(pred)

                all_f1 = 0
                for id_asp in range(len(ASPECT)):
                    tr = np.concatenate(true_label_list[idx2asp[id_asp]])
                    pr = np.concatenate(pred_label_list[idx2asp[id_asp]])
                    _, _, f1 = macro_f1(tr, pr)
                    all_f1 += f1
                
                avg_f1 = all_f1 / len(ASPECT)
                logger.info(f"  Dev Macro-F1: {avg_f1}")

                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    logger.info(f"  New Best F1 ({max_f1})! Saving best model...")
                    save_model(f'{args.output_dir}/seed_{args.seed}_fcmf_model_best.pth', model, optimizer, scheduler, train_idx, best_score=max_f1)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_best.pth', resnet_img, optimizer, scheduler, train_idx)
                    save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_best.pth', resnet_roi, optimizer, scheduler, train_idx)

                save_model(f'{args.output_dir}/seed_{args.seed}_fcmf_model_last.pth', model, optimizer, scheduler, train_idx, best_score=max_f1)
                save_model(f'{args.output_dir}/seed_{args.seed}_resimg_model_last.pth', resnet_img, optimizer, scheduler, train_idx)
                save_model(f'{args.output_dir}/seed_{args.seed}_resroi_model_last.pth', resnet_roi, optimizer, scheduler, train_idx)

    # ==========================================================================================
    # 7. TEST EVALUATION (With Detailed Logging)
    # ==========================================================================================
    if args.do_eval and master_process:
        logger.info("\n\n===================== STARTING TEST EVALUATION =====================")
        
        test_data = pd.read_json(f'{args.data_dir}/test.json')
        test_data['comment'] = test_data['comment'].apply(lambda x: normalize_class.normalize(text_normalize(convert_unicode(x))))
        test_dataset = MACSADataset(test_data, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.eval_batch_size)

        best_path = f'{args.output_dir}/seed_{args.seed}_fcmf_model_best.pth'
        if os.path.exists(best_path):
            logger.info(f"Loading Best Checkpoint from: {best_path}")
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            if isinstance(model, (DDP, torch.nn.DataParallel)):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load ResNets
            rimg_path = best_path.replace("fcmf", "resimg")
            if os.path.exists(rimg_path):
                rimg_ckpt = torch.load(rimg_path, map_location=device)
                unwrap_rimg = resnet_img.module if hasattr(resnet_img, 'module') else resnet_img
                unwrap_rimg.load_state_dict(rimg_ckpt['model_state_dict'])
                
            rroi_path = best_path.replace("fcmf", "resroi")
            if os.path.exists(rroi_path):
                rroi_ckpt = torch.load(rroi_path, map_location=device)
                unwrap_rroi = resnet_roi.module if hasattr(resnet_roi, 'module') else resnet_roi
                unwrap_rroi.load_state_dict(rroi_ckpt['model_state_dict'])
        else:
            logger.warning("No best model found! Using current weights.")

        model.eval(); resnet_img.eval(); resnet_roi.eval()

        true_label_list = {asp:[] for asp in ASPECT}
        pred_label_list = {asp:[] for asp in ASPECT}
        
        # [NEW] List to store results for formatted log
        formatted_results = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating Test"):
                batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                # Unpack bao gồm text
                (t_img_features, roi_img_features, roi_coors, 
                all_input_ids, all_token_types_ids, all_attn_mask, 
                all_added_input_mask, all_label_id, batch_texts) = batch

                # [CRITICAL] Fix DoubleTensor error in Test
                roi_img_features = roi_img_features.float()

                encoded_img = []
                for img_idx in range(args.num_imgs):
                    img_f = resnet_img(t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1)
                    encoded_img.append(img_f)
                encoded_roi = []
                for img_idx in range(args.num_imgs):
                    roi_list = [resnet_roi(roi_img_features[:,img_idx,r,:]).squeeze(1) for r in range(args.num_rois)]
                    encoded_roi.append(torch.stack(roi_list, dim=1))
                vis_embeds = torch.stack(encoded_img, dim=1)
                roi_embeds = torch.stack(encoded_roi, dim=1)

                # Initialize logs for current batch
                batch_logs = [{"text": t, "aspects": {}} for t in batch_texts]

                for id_asp in range(len(ASPECT)):
                    aspect_name = ASPECT[id_asp]
                    
                    logits = model(
                        input_ids=all_input_ids[:,id_asp,:],
                        token_type_ids=all_token_types_ids[:,id_asp,:],
                        attention_mask=all_attn_mask[:,id_asp,:],
                        added_attention_mask=all_added_input_mask[:,id_asp,:],
                        visual_embeds_att=vis_embeds,
                        roi_embeds_att=roi_embeds,
                        roi_coors=roi_coors
                    )
                    preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                    labels = all_label_id[:,id_asp].cpu().numpy()
                    
                    true_label_list[aspect_name].append(labels)
                    pred_label_list[aspect_name].append(preds)
                    
                    # Store detailed result
                    for i, (p, l) in enumerate(zip(preds, labels)):
                        batch_logs[i]["aspects"][aspect_name] = {
                            "predict": POLARITY_MAP.get(p, "Unknown"),
                            "label": POLARITY_MAP.get(l, "Unknown")
                        }
                
                formatted_results.extend(batch_logs)

        # 1. Calculate & Save Metrics
        output_eval_file = os.path.join(args.output_dir, "test_results_fcmf.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Test results *****\n")
            all_f1 = 0
            for id_asp in range(len(ASPECT)):
                tr = np.concatenate(true_label_list[ASPECT[id_asp]])
                pr = np.concatenate(pred_label_list[ASPECT[id_asp]])
                precision, recall, f1 = macro_f1(tr, pr)
                all_f1 += f1
                writer.write(f"{ASPECT[id_asp]} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}\n")
                logger.info(f"{ASPECT[id_asp]} - F1: {f1:.4f}")
            
            avg_f1 = all_f1 / len(ASPECT)
            writer.write(f"Average F1: {avg_f1:.4f}\n")
            logger.info(f"Average F1: {avg_f1:.4f}")

        # 2. Save Formatted Detailed Log
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