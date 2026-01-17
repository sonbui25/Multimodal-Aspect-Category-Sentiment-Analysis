import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torchvision.transforms as T
import argparse
import logging
from torch.utils.data import Dataset, DataLoader

# --- CẤU HÌNH LOGGER ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        # Lọc file ảnh hợp lệ
        self.image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, file_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, file_name
        except Exception as e:
            return torch.zeros(3, 299, 299), "error"

def create_caption_and_mask(start_token, max_length, batch_size):
    caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../vimacsa/image', type=str)
    parser.add_argument("--output_file", default='visual_captions_catr_vi.json', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    # [FIX] Mặc định max_len phải là 128 theo kiến trúc CATR
    parser.add_argument("--max_len", default=128, type=int) 
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    device = args.device
    logger.info(f"Running on {device} with Batch Size {args.batch_size}...")

    # 1. LOAD MODEL CATR
    logger.info("Loading CATR model (v3)...")
    catr_model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    catr_model.to(device)
    catr_model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    catr_transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. LOAD TRANSLATOR
    logger.info("Loading Translator (EnViT5)...")
    trans_model_name = "VietAI/envit5-translation"
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(device)
    trans_model.eval()

    def translate_batch(texts):
        valid_indices = [i for i, t in enumerate(texts) if t]
        if not valid_indices: return texts
        
        inputs = [f"en: {texts[i]}" for i in valid_indices]
        encoded = trans_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            generated_ids = trans_model.generate(**encoded, max_length=128)
        decoded = trans_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        results = list(texts)
        for i, idx in enumerate(valid_indices):
            results[idx] = decoded[i].replace("vi: ", "").strip()
        return results

    # 3. DATALOADER
    dataset = ImageDataset(args.image_dir, catr_transform)
    # num_workers để load ảnh song song
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=18)

    captions_dict = {}

    # 4. GENERATION LOOP
    logger.info("Starting Batch Generation...")
    
    for images, filenames in tqdm(dataloader, desc="Generating"):
        current_batch_size = images.size(0)
        images = images.to(device)
        
        caption, cap_mask = create_caption_and_mask(start_token, args.max_len, current_batch_size)
        caption = caption.to(device)
        cap_mask = cap_mask.to(device)
        
        # Theo dõi trạng thái hoàn thành của từng ảnh trong batch
        finished = torch.zeros(current_batch_size, dtype=torch.bool).to(device)

        with torch.no_grad():
            for step in range(args.max_len - 1):
                predictions = catr_model(images, caption, cap_mask)
                predictions = predictions[:, step, :]
                predicted_ids = torch.argmax(predictions, axis=-1)
                
                # Gán token dự đoán
                caption[:, step+1] = predicted_ids
                cap_mask[:, step+1] = False
                
                # Cập nhật trạng thái 'finished' nếu gặp token SEP
                finished |= (predicted_ids == end_token)
                
                # [FIX] Dừng sớm nếu TOÀN BỘ batch đã xong
                if finished.all():
                    break

        # Decode Batch
        en_captions = []
        for i in range(current_batch_size):
            if filenames[i] == "error":
                en_captions.append("")
                continue
                
            output_ids = caption[i].tolist()
            try:
                # Cắt chuỗi tại SEP token đầu tiên
                end_idx = output_ids.index(end_token)
                output_ids = output_ids[1:end_idx]
            except ValueError:
                output_ids = output_ids[1:] # Lấy hết nếu ko thấy SEP
            
            cap_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            en_captions.append(cap_text)

        # Translate Batch
        vi_captions = translate_batch(en_captions)

        for fname, vicap in zip(filenames, vi_captions):
            if fname != "error":
                captions_dict[fname] = vicap
                print(f"{fname}: {vicap}")
    # 5. SAVE
    output_path = args.output_file
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)
    logger.info(f"Done! Saved {len(captions_dict)} captions to {output_path}")

if __name__ == "__main__":
    main()