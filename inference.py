from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import torch
import torchvision
import numpy as np
from torchvision.transforms import v2
from ultralytics import YOLO
import yaml
import cv2 
from transformers import AutoTokenizer
from underthesea import word_tokenize,text_normalize
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from fcmf_framework.fcmf_multimodal import FCMF
from text_preprocess import *
from fcmf_framework.image_process import *
import argparse
from loguru import logger
import sys
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
IMG_ASPECT = ['Food', 'Room', 'Facilities', 'Service', 'Public_area']
POLARITY = ['None','Negative','Neutral','Positive']
BASE_PATH = 'checkpoints_ViIM_FCMF'
YOLO_PATH = './checkpoints_yolo/yolov8m.pt'
WEIGHT_ROI_PATH = f'./{BASE_PATH}/seed_42_resroi_model_best.pth'
WEIGHT_IMAGE_PATH = f'./{BASE_PATH}/seed_42_resimg_model_best.pth'
FCMF_CHECKPOINT = f'./{BASE_PATH}/seed_42_fcmf_model_best.pth'
VISUAL_MODEL_CHECKPOINT = './4_visual_model.pth'

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
# logger.add("file_{time}.log")  # Disabled - use --output_file instead,

# =============================== CHECK FILE PATH ===============================
def check_file_exists(path):
    """Check if file exists, return absolute path if yes, else None"""
    import os
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        logger.info(f"✓ Found: {abs_path}")
        return abs_path
    else:
        logger.error(f"✗ File not found: {abs_path}")
        return None

# =============================== LOAD ROI MODEL ===============================
def load_yolo_roi_model(yolo_path,weight_roi_path):
    roi_model = MyRoIModel(len(IMG_ASPECT)) # No Location
    roi_model = roi_model.to(device)

    # Check YOLO path
    yolo_path = check_file_exists(yolo_path)
    if yolo_path is None:
        logger.error(f"YOLO model not found!")
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")

    yolo_model = YOLO(yolo_path)
    # Set YOLO to CPU - torchvision NMS doesn't support CUDA
    yolo_model.to('cpu')
    logger.info("YOLO model set to CPU (torchvision NMS compatibility)")

    # Check ROI weight path
    weight_roi_path = check_file_exists(weight_roi_path)
    if weight_roi_path is None:
        logger.error(f"RoI model not found!")
        raise FileNotFoundError(f"RoI model not found at {weight_roi_path}")

    try:
        checkpoint = load_model(weight_roi_path)
        state_dict = checkpoint['model_state_dict']
        
        # Fix key mismatch: rename 'resnet.*' to 'feature_extractor.*'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('resnet.'):
                # Replace 'resnet.' with 'feature_extractor.'
                new_key = key.replace('resnet.', 'feature_extractor.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        logger.info(f"Loading RoI model with {len(new_state_dict)} keys (renamed resnet.* -> feature_extractor.*)")
        roi_model.load_state_dict(new_state_dict, strict=False)
            
    except Exception as e:
        logger.error(f"Error loading RoI checkpoint: {e}")
        raise ValueError(f"Wrong RoI weight path or corrupted file: {e}")
    
    roi_model.eval()

    return yolo_model, roi_model

# =============================== LOAD IMAGE MODEL ===============================
def load_image_model(weight_image_path):
    image_model = MyImgModel(len(IMG_ASPECT)) # No Location
    image_model = image_model.to(device)

    # Check image weight path
    weight_image_path = check_file_exists(weight_image_path)
    if weight_image_path is None:
        logger.error(f"Image model not found!")
        raise FileNotFoundError(f"Image model not found at {weight_image_path}")

    try:
        checkpoint = load_model(weight_image_path)
        state_dict = checkpoint['model_state_dict']
        
        # Fix key mismatch: rename 'resnet.*' to 'feature_extractor.*'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('resnet.'):
                # Replace 'resnet.' with 'feature_extractor.'
                new_key = key.replace('resnet.', 'feature_extractor.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        logger.info(f"Loading Image model with {len(new_state_dict)} keys (renamed resnet.* -> feature_extractor.*)")
        image_model.load_state_dict(new_state_dict, strict=False)
            
    except Exception as e:
        logger.error(f"Error loading Image checkpoint: {e}")
        raise ValueError(f"Wrong Image weight path or corrupted file: {e}")

    image_model.eval()

    return image_model

# ============================  LOADING TRAINED MODEL ============================ 
def load_fcmf_model(fcmf_checkpoint,visual_checkpoint, pretrained_model, num_imgs, num_rois, tokenizer):
    fcmf_model = FCMF(pretrained_model,num_imgs=num_imgs,num_roi=num_rois)
    fcmf_model.to(device)
    
    # Resize token embeddings to match tokenizer vocab size
    fcmf_model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized token embeddings to {len(tokenizer)}")

    visual_model = resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
    visual_model.to(device)
    
    # Check FCMF checkpoint path
    fcmf_checkpoint = check_file_exists(fcmf_checkpoint)
    if fcmf_checkpoint is None:
        logger.error(f"FCMF model not found!")
        raise FileNotFoundError(f"FCMF model not found at {fcmf_checkpoint}")
    
    try:
        fcmf_ckpt_data = load_model(fcmf_checkpoint)
        # Try to load with strict=False
        try:
            result = fcmf_model.load_state_dict(fcmf_ckpt_data['model_state_dict'], strict=False)
            
            if result.missing_keys:
                logger.warning(f"FCMF Missing keys ({len(result.missing_keys)}): {result.missing_keys[:3]}...")
            if result.unexpected_keys:
                logger.warning(f"FCMF Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:3]}...")
            
            logger.info("✓ FCMF model loaded successfully")
        except RuntimeError as e:
            # If size mismatch, try to skip embedding layer
            if "size mismatch" in str(e):
                logger.warning(f"Size mismatch in embeddings: {e}")
                logger.warning("Attempting to load non-embedding layers...")
                
                state_dict = fcmf_ckpt_data['model_state_dict']
                # Remove embedding layers that have size mismatch
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if 'word_embeddings.weight' not in k}
                
                result = fcmf_model.load_state_dict(filtered_state_dict, strict=False)
                logger.warning(f"Loaded {len(filtered_state_dict)} keys (skipped embedding layer)")
                logger.info("✓ FCMF model loaded with filtered state dict")
            else:
                raise e
                
    except ValueError as e:
        logger.error(f"Error loading FCMF checkpoint: {e}")
        raise ValueError(f"Wrong FCMF weight path or corrupted file: {e}")
    except Exception as e:
        logger.error(f"Error loading FCMF checkpoint: {e}")
        raise ValueError(f"Wrong FCMF weight path or corrupted file: {e}")
    
    # Try to load visual checkpoint, fallback to pretrained if not found
    visual_checkpoint = check_file_exists(visual_checkpoint)
    if visual_checkpoint is None:
        logger.warning(f"Visual model checkpoint not found")
        logger.warning("Using ResNet152 with pretrained ImageNet weights instead")
    else:
        try:
            visual_checkpoint_data = load_model(visual_checkpoint)
            visual_model.load_state_dict(visual_checkpoint_data['model_state_dict'])
            logger.info(f"✓ Loaded visual model from: {visual_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load visual model checkpoint: {e}")
            logger.warning("Using ResNet152 with pretrained ImageNet weights instead")
    
    return fcmf_model, visual_model

# ============================  GETTING VISUAL FEATURES ============================ 
def get_visual_features(yolo_model, visual_model, list_image_path, num_imgs, num_rois, device):
    t_img_features, roi_img_features, roi_coors = construct_visual_features(yolo_model,list_image_path, 30, num_rois, num_imgs, device)
    t_img_features = t_img_features.unsqueeze(0)
    t_img_features = t_img_features.float().to(device)
    roi_img_features = roi_img_features.unsqueeze(0)
    roi_img_features = roi_img_features.float().to(device)
    roi_coors = roi_coors.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_img = []
        encoded_roi = []

        for img_idx in range(num_imgs):
            img_features = image_encoder(visual_model,t_img_features[:,img_idx,:]).view(-1,2048,49).permute(0,2,1).squeeze(1) # batch_size, 49, 2048
            encoded_img.append(img_features)

            roi_f = []
            for roi_idx in range(num_rois):
                roi_features = roi_encoder(visual_model,roi_img_features[:,img_idx,roi_idx,:]).squeeze(1) # batch_size, 1, 2048
                roi_f.append(roi_features)

            roi_f = torch.stack(roi_f,dim=1)
            encoded_roi.append(roi_f)

        encoded_img = torch.stack(encoded_img,dim=1) # batch_size, num_img, 49, 2048   
        encoded_roi = torch.stack(encoded_roi,dim=1) # batch_size, num_img, num_roi, 49,2048

    return roi_coors, encoded_img, encoded_roi

# ============================  FCMF PREDICTION ============================ 
def fcmf_predict_wrapper(tokenizer, text, IMG_ASPECT, ASPECT, list_image_path, num_imgs, num_rois, device):
    # ====== ASPECT PREDICTION ======
    print("============ LOADING MODEL ============")
    logger.info("loading model")
    yolo_model, roi_model = load_yolo_roi_model(YOLO_PATH, WEIGHT_ROI_PATH)
    image_model = load_image_model(WEIGHT_IMAGE_PATH)
    fcmf_model, visual_model = load_fcmf_model(FCMF_CHECKPOINT, VISUAL_MODEL_CHECKPOINT, pretrained_model, num_imgs, num_rois, tokenizer)

    logger.info('construct features')
    print("============ CONSTRUCT FEATURES ============")
    list_image_aspect, list_roi_aspect = image_processing(image_model,roi_model, yolo_model, list_image_path, 30, IMG_ASPECT, device)
    joined_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}" # RIGHT PATH FOR AUXILIARY SENTENCE
    joined_aspect = joined_aspect.lower().replace('_',' ')

    # VISUAL FEATURES
    roi_coors, encoded_img, encoded_roi = get_visual_features(yolo_model, visual_model, list_image_path,num_imgs, num_rois, device)
    
    logger.info('making prediction')
    print("============ MAKING PREDICTION ============")
    rs = {asp:'None' for asp in ASPECT}
    for id_asp in range(len(ASPECT)):
        asp = ASPECT[id_asp]
        combine_text = f"{asp} </s></s> {text}"
        combine_text = combine_text.lower().replace('_',' ')
        tokens = tokenizer(combine_text, joined_aspect, max_length=170,truncation='only_first',padding='max_length', return_token_type_ids=True)

        input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(tokens['token_type_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device)
        added_input_mask =torch.tensor( [1] * (170+49)).unsqueeze(0).to(device)

        logits = fcmf_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, 
            added_attention_mask = added_input_mask,
            visual_embeds_att = encoded_img,
            roi_embeds_att = encoded_roi,
            roi_coors = roi_coors
        )
        pred = np.argmax(logits.detach().cpu(),axis = -1)

        rs[ASPECT[id_asp]] = POLARITY[pred[0]]

    logger.success("Done")
    logger.success(f'{rs}')
    return rs
# ================== TEXT ==================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",
                    type=str,
                    required=True,
                    help="text input")

    parser.add_argument("--image_list", 
                        "--names-list", 
                        nargs='+', help = "Optional. List of image associated with text.")

    parser.add_argument("--num_images",
                        type=int,
                        default = 7,
                        required=False,
                        help="number of images")

    parser.add_argument("--num_rois",
                        type=int,
                        default = 4,
                        required=False,
                        help="number of RoIs")

    parser.add_argument("--pretrained_model",
                        type=str,
                        required=False,
                        default='xlm-roberta-base',
                        help="pretrained model for FCMF framework")
    
    parser.add_argument("--output_file",
                        type=str,
                        required=False,
                        default=None,
                        help="Optional. Path to save prediction results to file")

    args = parser.parse_args()

    num_rois = args.num_rois
    num_imgs = args.num_images
    list_image_path = args.image_list
    if list_image_path == None:
        list_image_path = []
    else:
        # Fix path issues: convert escaped backslashes and handle absolute paths
        fixed_paths = []
        for p in list_image_path:
            # Convert forward slashes to backslashes for Windows
            p = p.replace('/', '\\')
            # If path starts with drive letter (e.g., E:), it's absolute - use as is
            if len(p) > 1 and p[1] == ':':
                fixed_paths.append(p)
            else:
                # Otherwise treat as relative to current directory
                fixed_paths.append(os.path.abspath(p))
        list_image_path = fixed_paths
        
        logger.info(f"Image paths: {list_image_path}")
    
    text = args.text
    normalize_class = TextNormalize()
    text = normalize_class.normalize(text_normalize(convert_unicode(text)))    

    pretrained_model = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    print(f"Using {num_imgs} images and {num_rois} RoIs.")
    logger.info(f"Using {num_imgs} images and {num_rois} RoIs.")
    print(f"Using {pretrained_model} for text features extraction.")
    logger.info(f"Using {pretrained_model} for text features extraction.")

    result = fcmf_predict_wrapper(
                tokenizer = tokenizer,\
                text = text, \
                IMG_ASPECT = IMG_ASPECT, \
                ASPECT = ASPECT, \
                list_image_path = list_image_path, \
                num_imgs = num_imgs, \
                num_rois = num_rois, \
                device = device
            )
    print(result)
    
    # Save result to file if output_file is specified
    if args.output_file:
        output_path = args.output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Text: {text}\n")
            f.write(f"Number of images: {len(list_image_path)}\n")
            if list_image_path:
                f.write(f"Images: {', '.join(list_image_path)}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("PREDICTIONS:\n")
            f.write("="*50 + "\n\n")
            for aspect, polarity in result.items():
                f.write(f"{aspect}: {polarity}\n")
        print(f"\n✓ Results saved to: {output_path}")
        logger.success(f"Results saved to: {output_path}")
    
