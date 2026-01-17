import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torchvision.transforms as T
import argparse
import logging

# --- CẤU HÌNH LOGGER ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../vimacsa/image', type=str, help="Folder ảnh")
    parser.add_argument("--output_file", default='visual_captions_catr_vi.json', type=str)
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    device = args.device
    logger.info(f"Running on {device}...")

    # ==============================================================================
    # 1. LOAD MODEL CATR (ORIGINAL PAPER MODEL)
    # ==============================================================================
    logger.info("Loading CATR model (v3) from torch.hub...")
    catr_model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    catr_model.to(device)
    catr_model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # [ĐÃ SỬA LỖI Ở ĐÂY]
    # Thay tokenizer._cls_token thành tokenizer.cls_token
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    catr_transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==============================================================================
    # 2. LOAD TRANSLATOR (EnViT5)
    # ==============================================================================
    logger.info("Loading Translator (EnViT5)...")
    trans_model_name = "VietAI/envit5-translation"
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(device)
    trans_model.eval()

    def translate_en_to_vi(texts):
        inputs = [f"en: {t}" for t in texts]
        encoded = trans_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            generated_ids = trans_model.generate(**encoded, max_length=128)
        decoded = trans_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.replace("vi: ", "").strip() for t in decoded]

    # ==============================================================================
    # 3. GENERATE LOOP
    # ==============================================================================
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    # Kiểm tra đường dẫn tồn tại trước khi list
    if not os.path.exists(args.image_dir):
        logger.error(f"Image directory not found: {args.image_dir}")
        return

    image_files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    logger.info(f"Found {len(image_files)} images.")

    captions_dict = {}
    caption_buffer = [] 
    buffer_size = 16 

    for i, file_name in enumerate(tqdm(image_files, desc="Generating (CATR)")):
        try:
            img_path = os.path.join(args.image_dir, file_name)
            image = Image.open(img_path).convert('RGB')
            image_tensor = catr_transform(image).unsqueeze(0).to(device)

            # --- CATR Inference ---
            caption, cap_mask = create_caption_and_mask(start_token, max_length=128)
            caption = caption.to(device)
            cap_mask = cap_mask.to(device)

            with torch.no_grad():
                for step in range(128 - 1):
                    predictions = catr_model(image_tensor, caption, cap_mask)
                    predictions = predictions[:, step, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    
                    if predicted_id[0] == end_token:
                        break
                    
                    caption[:, step+1] = predicted_id[0]
                    cap_mask[:, step+1] = False

            output_ids = caption[0].tolist()
            try:
                # Tìm vị trí token kết thúc để cắt chuỗi
                end_idx = output_ids.index(end_token)
                output_ids = output_ids[1:end_idx] 
            except ValueError:
                output_ids = output_ids[1:] 
            
            en_caption = tokenizer.decode(output_ids, skip_special_tokens=True)
            caption_buffer.append((file_name, en_caption))

            # --- Translate Batch ---
            if len(caption_buffer) >= buffer_size or i == len(image_files) - 1:
                filenames, en_texts = zip(*caption_buffer)
                vi_texts = translate_en_to_vi(en_texts)
                
                for fname, vicap in zip(filenames, vi_texts):
                    captions_dict[fname] = vicap
                
                caption_buffer = []

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

    # ==============================================================================
    # 4. SAVE RESULT
    # ==============================================================================
    # Xử lý đường dẫn file output an toàn
    if os.path.dirname(args.output_file):
        output_path = args.output_file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        # Nếu chỉ đưa tên file, lưu vào thư mục làm việc hiện tại hoặc image_dir
        output_path = args.output_file

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Done! Saved CATR captions to {output_path}")

if __name__ == "__main__":
    main()