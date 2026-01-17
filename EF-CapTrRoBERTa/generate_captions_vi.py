import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=r'E:\DS310\ViMACSA\ViMACSA\image', type=str)
    parser.add_argument("--output_file", default=r'E:\DS310\ViMACSA\ViMACSA\visual_captions_vi.json', type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # 1. Load Model Caption (BLIP) - Tiếng Anh
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model_cap.eval()

    # 2. Load Model Dịch (Helsinki-NLP) - Anh sang Việt
    print("Loading Translation model (En -> Vi)...")
    trans_model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer_trans = MarianTokenizer.from_pretrained(trans_model_name)
    model_trans = MarianMTModel.from_pretrained(trans_model_name).to(device)
    model_trans.eval()

    # 3. Lấy danh sách ảnh
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    captions_dict = {}

    # 4. Vòng lặp xử lý (Batch)
    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Generating Vietnamese Captions"):
        batch_files = image_files[i : i + args.batch_size]
        batch_images = []
        valid_batch_files = []

        # Load ảnh
        for file_name in batch_files:
            try:
                img_path = os.path.join(args.image_dir, file_name)
                image = Image.open(img_path).convert('RGB')
                batch_images.append(image)
                valid_batch_files.append(file_name)
            except Exception as e:
                print(f"Error: {e}")

        if not batch_images: continue

        # A. Sinh Caption Tiếng Anh
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_cap.generate(**inputs, max_new_tokens=50, num_beams=5)
        en_captions = processor.batch_decode(out, skip_special_tokens=True)
        print(en_captions)
        # B. Dịch sang Tiếng Việt
        # Tokenize batch tiếng Anh
        batch_inputs_trans = tokenizer_trans(en_captions, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model_trans.generate(**batch_inputs_trans)
        vi_captions = tokenizer_trans.batch_decode(translated, skip_special_tokens=True)
        print(vi_captions)
        # C. Lưu kết quả
        for file_name, vi_cap in zip(valid_batch_files, vi_captions):
            captions_dict[file_name] = vi_cap

    # 5. Lưu file JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Hoàn tất! Caption tiếng Việt đã lưu tại {args.output_file}")

if __name__ == "__main__":
    main()