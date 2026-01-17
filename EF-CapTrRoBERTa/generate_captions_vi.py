import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../vimacsa/image', type=str)
    parser.add_argument("--output_file", default='../vimacsa/visual_captions_vi.json', type=str)
    parser.add_argument("--batch_size", default=8, type=int) # EnViT5 hơi nặng hơn Helsinki chút, giảm batch nếu cần
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # 1. Load Model Caption (BLIP)
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model_cap.eval()

    # 2. Load Model Dịch (VietAI/envit5-translation)
    print("Loading Translation model (VietAI EnViT5)...")
    trans_model_name = "VietAI/envit5-translation"
    tokenizer_trans = AutoTokenizer.from_pretrained(trans_model_name)
    model_trans = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name).to(device)
    model_trans.eval()

    # Hàm hỗ trợ dịch batch với EnViT5
    def translate_batch(texts):
        # EnViT5 yêu cầu prefix "en: " đầu câu để biết chiều dịch
        inputs = [f"en: {text}" for text in texts]
        encoded = tokenizer_trans(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generated_ids = model_trans.generate(**encoded, max_length=512)
        decoded = tokenizer_trans.batch_decode(generated_ids, skip_special_tokens=True)
        # Kết quả trả về thường có dạng "vi: nội dung", cần cắt bỏ "vi: "
        return [text.replace("vi: ", "").strip() for text in decoded]

    # 3. Lấy danh sách ảnh
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in os.listdir(args.image_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    captions_dict = {}

    # 4. Vòng lặp xử lý
    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Generating Captions"):
        batch_files = image_files[i : i + args.batch_size]
        batch_images = []
        valid_batch_files = []

        for file_name in batch_files:
            try:
                img_path = os.path.join(args.image_dir, file_name)
                image = Image.open(img_path).convert('RGB')
                batch_images.append(image)
                valid_batch_files.append(file_name)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

        if not batch_images: continue

        # A. Sinh Caption Tiếng Anh (BLIP)
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_cap.generate(**inputs, max_new_tokens=50, num_beams=5)
        en_captions = processor.batch_decode(out, skip_special_tokens=True)

        # B. Dịch sang Tiếng Việt (EnViT5)
        vi_captions = translate_batch(en_captions)

        # C. Lưu kết quả
        for file_name, vi_cap in zip(valid_batch_files, vi_captions):
            captions_dict[file_name] = vi_cap
            print(f"{file_name}: {vi_cap}")

    # 5. Lưu file JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Hoàn tất! Caption đã lưu tại {args.output_file}")

if __name__ == "__main__":
    main()