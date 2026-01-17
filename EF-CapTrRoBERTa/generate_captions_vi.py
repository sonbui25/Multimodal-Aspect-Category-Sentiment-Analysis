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

# --- CẤU HÌNH LOGGER ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. CÁC CLASS HỖ TRỢ (GIỐNG HỆT CODE GỐC) ---

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

# Transform chuẩn của CATR (Thay thế cho coco.val_transform)
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

# --- 2. HÀM EVALUATE (GIỮ NGUYÊN LOGIC AUTOREGRESSIVE) ---
@torch.no_grad()
def evaluate(model, image, caption, cap_mask, max_pos_emb):
    model.eval()
    # Vòng lặp sinh từ (Đây là cách code gốc hoạt động)
    for i in range(max_pos_emb - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102: # Token SEP (Kết thúc)
            return caption

        caption[:, i+1] = predicted_id[0]
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
            
            # Quy trình xử lý ảnh gốc: SquarePad -> Resize 299 -> Transform
            image = self.sqpad(image)
            image = F.resize(image, 299)
            image = self.transform(image)
            
            # CATR nhận input shape [1, 3, 299, 299] (thêm batch dimension ảo nếu chạy đơn lẻ)
            # Nhưng DataLoader sẽ tự stack, nên ở đây trả về [3, 299, 299]
            
            caption, cap_mask = create_caption_and_mask(
                self.start_token, self.max_position_embeddings)

            return {
                "image": image,
                "caption": caption,
                "cap_mask": cap_mask,
                "file_name": os.path.basename(img_path)
            }
        except Exception as e:
            return None

def main():
    parser = argparse.ArgumentParser(description='Image Captioning Original Style')
    parser.add_argument('--image_dir', type=str, default='../vimacsa/image', help='path to images')
    parser.add_argument('--output_file', type=str, default='visual_captions_catr_original_en.json')
    parser.add_argument('--batch_size', type=int, default=1, help="Code gốc chạy từng ảnh, nhưng ta có thể batch để nhanh hơn nếu muốn. Để 1 cho giống hệt.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}...")

    # Load Model (v3 như file gốc)
    logger.info("Loading CATR v3...")
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    model.to(device)
    model.eval()

    # Tokenizer & Config
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = Config()
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token) # dùng cls_token thay vì _cls_token để tránh lỗi version

    # Dataset & Loader
    dataset = TwitterImagesDataset(args.image_dir, start_token, config.max_position_embeddings)
    # Code gốc chạy loop thuần túy, ở đây dùng DataLoader để quản lý file tốt hơn
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=18, collate_fn=lambda x: x[0] if x[0] is not None else None)

    img_to_caption = {}
    errors = 0

    logger.info("Starting Generation (Original Autoregressive Loop)...")
    for batch in tqdm(dataloader):
        if batch is None:
            errors += 1
            continue

        image = batch['image'].unsqueeze(0).to(device) # Thêm batch dim [1, 3, 299, 299]
        caption = batch['caption'].to(device)
        cap_mask = batch['cap_mask'].to(device)
        file_name = batch['file_name']

        try:
            # Gọi hàm evaluate y hệt file gốc
            output = evaluate(model, image, caption, cap_mask, config.max_position_embeddings)
            
            # Decode
            result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            result = result.capitalize()
            
            img_to_caption[file_name] = result
            print(f"{file_name}: {result}")
        except Exception as e:
            errors += 1
            # print(f"Error processing {file_name}: {e}")

    # Save
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(img_to_caption, f, indent=4)
    
    logger.info(f"Done! Saved {len(img_to_caption)} captions. Errors: {errors}")

if __name__ == "__main__":
    main()