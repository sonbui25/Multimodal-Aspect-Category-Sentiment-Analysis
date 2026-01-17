import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer
from PIL import Image
import argparse
import os
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. UTILS (GIỮ NGUYÊN) ---
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

class Config:
    def __init__(self):
        self.max_position_embeddings = 128
        self.hidden_dim = 768

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_caption_and_mask(start_token, max_length):
    # Hàm này tạo caption cho 1 ảnh, ta sẽ dùng repeat trong dataset hoặc tạo batch trong loop
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

# --- 2. EVALUATE (Hỗ trợ Batch) ---
@torch.no_grad()
def evaluate(model, image, caption, cap_mask, max_pos_emb):
    model.eval()
    # Vòng lặp Autoregressive (Sinh từng từ)
    for i in range(max_pos_emb - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        # Gán token dự đoán vào bước tiếp theo
        caption[:, i+1] = predicted_id
        cap_mask[:, i+1] = False

    return caption

# --- 3. DATASET ---
class TwitterImagesDataset(Dataset):
    def __init__(self, image_dir, start_token, max_position_embeddings):
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if os.path.splitext(f)[1].lower() in valid_extensions]
        self.start_token = start_token
        self.max_position_embeddings = max_position_embeddings
        self.sqpad = SquarePad()
        self.transform = get_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.sqpad(image)
            image = F.resize(image, 299)
            image = self.transform(image)
            
            # Tạo template caption [1, 128] -> squeeze thành [128] để DataLoader tự stack
            caption, cap_mask = create_caption_and_mask(
                self.start_token, self.max_position_embeddings)
            
            return {
                "image": image,
                "caption": caption.squeeze(0), 
                "cap_mask": cap_mask.squeeze(0),
                "file_name": os.path.basename(img_path),
                "valid": True
            }
        except Exception as e:
            # Trả về dummy data nếu lỗi (để không làm crash dataloader)
            return {
                "image": torch.zeros(3, 299, 299),
                "caption": torch.zeros(128, dtype=torch.long),
                "cap_mask": torch.ones(128, dtype=torch.bool),
                "file_name": "error",
                "valid": False
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='../vimacsa/image')
    parser.add_argument('--output_file', type=str, default='visual_captions_catr_original_en.json')
    parser.add_argument('--batch_size', type=int, default=32) # Batch size lớn để chạy nhanh
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device} with Batch Size {args.batch_size}...")

    # Load Model
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = Config()
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)

    # Dataset & Loader
    dataset = TwitterImagesDataset(args.image_dir, start_token, config.max_position_embeddings)
    
    # [QUAN TRỌNG] Bỏ collate_fn tùy chỉnh, dùng mặc định để stack batch
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    img_to_caption = {}

    for batch in tqdm(dataloader, desc="Generating"):
        # Lọc bỏ các mẫu lỗi (nếu có)
        valid_mask = batch['valid']
        if not valid_mask.any(): continue
        
        images = batch['image'][valid_mask].to(device)
        captions = batch['caption'][valid_mask].to(device)
        cap_masks = batch['cap_mask'][valid_mask].to(device)
        filenames = np.array(batch['file_name'])[valid_mask.cpu().numpy()]

        # Chạy model (Batch processing)
        # Hàm evaluate này vẫn giữ logic Autoregressive gốc nhưng chạy song song nhiều ảnh
        outputs = evaluate(model, images, captions, cap_masks, config.max_position_embeddings)

        # Decode kết quả
        for i, output_ids in enumerate(outputs):
            # Tìm token SEP (102) để cắt chuỗi
            out_list = output_ids.tolist()
            if 102 in out_list:
                end_idx = out_list.index(102)
                out_list = out_list[:end_idx]
            
            result = tokenizer.decode(out_list, skip_special_tokens=True)
            result = result.capitalize()
            img_to_caption[filenames[i]] = result

    # Save
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(img_to_caption, f, indent=4)
    
    logger.info(f"Done! Saved {len(img_to_caption)} captions.")

if __name__ == "__main__":
    main()