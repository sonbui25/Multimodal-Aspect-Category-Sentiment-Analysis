import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer
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

def create_paper_prompt(start_token, max_length, batch_size):
    # Tạo Prompt v* = [CLS, 0, 0...] cho One-Pass Generation
    caption = torch.zeros((batch_size, max_length), dtype=torch.long)
    caption[:, 0] = start_token
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool) # Unmask toàn bộ
    return caption, mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../vimacsa/image', type=str)
    parser.add_argument("--output_file", default='visual_captions_catr_en.json', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_len", default=128, type=int) # Cố định 128
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    device = args.device
    logger.info(f"Generating RAW ENGLISH Captions on {device}...")

    # 1. LOAD MODEL CATR (Chỉ load CATR, bỏ qua Translator)
    logger.info("Loading CATR model...")
    catr_model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    catr_model.to(device)
    catr_model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    
    catr_transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. DATALOADER
    dataset = ImageDataset(args.image_dir, catr_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    captions_dict = {}

    # 3. GENERATION (Không có bước dịch)
    for images, filenames in tqdm(dataloader, desc="Generating EN Captions"):
        current_batch_size = images.size(0)
        images = images.to(device)
        
        caption, cap_mask = create_paper_prompt(start_token, args.max_len, current_batch_size)
        caption = caption.to(device)
        cap_mask = cap_mask.to(device)

        with torch.no_grad():
            # One Forward Pass
            predictions = catr_model(images, caption, cap_mask)
            predicted_ids = torch.argmax(predictions, axis=-1)

        # Decode trực tiếp ra Tiếng Anh
        for i in range(current_batch_size):
            if filenames[i] == "error":
                continue
            
            output_ids = predicted_ids[i].tolist()
            # Decode thành text tiếng Anh
            cap_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Lưu nguyên bản (Raw English)
            captions_dict[filenames[i]] = cap_text.strip()
            print(filenames[i], "=>", cap_text.strip())
    # 4. SAVE
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Done! Saved RAW ENGLISH captions to {args.output_file}")

if __name__ == "__main__":
    main()