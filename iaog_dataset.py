import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np

class IAOGDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=7, max_len_decoder=20):
        """
        Args:
            max_len_decoder (int): Độ dài tối đa cho chuỗi sinh ra (bao gồm cả special tokens).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.img_folder = img_folder
        self.roi_df = roi_df
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi
        self.max_len_decoder = max_len_decoder

        # --- Transform Image ---
        # Chuẩn hóa ảnh về kích thước 224x224 và normalize theo ImageNet stats
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 1. Lấy dữ liệu thô
        idx_data = self.data.iloc[idx, :].values
        text = idx_data[0]       # Comment text
        list_img_path = idx_data[1] # List các đường dẫn ảnh
        iaog_label = idx_data[6]    # List các nhãn aspect-sentiment

        # ----------------------------------------------------------------------
        # PHẦN 1: XỬ LÝ TEXT INPUT CHO ENCODER (Gồm Comment + Các Tags từ ảnh)
        # ----------------------------------------------------------------------
        
        # Lấy aspect tags từ dictionary (Logic giữ nguyên từ code gốc)
        list_image_aspect = []
        list_roi_aspect = []
        for img_name in list_img_path[:self.num_img]:
            try:
                list_image_aspect.extend(self.dict_image_aspect[img_name])
            except:
                pass
            try:
                list_roi_aspect.extend(self.dict_roi_aspect[img_name])
            except:
                pass

        list_image_aspect = list(set(list_image_aspect))
        list_roi_aspect = list(set(list_roi_aspect))

        if len(list_image_aspect) == 0: list_image_aspect = ['empty']
        if len(list_roi_aspect) == 0: list_roi_aspect = ['empty']

        # Tạo chuỗi input phụ trợ từ tags
        joined_image_roi_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
        joined_image_roi_aspect = joined_image_roi_aspect.lower().replace('_', ' ')

        # Tạo input cho Encoder: [CLS] Comment [SEP] Tags [SEP]
        comment_text = f"{text}".lower().replace('_', ' ')
        
        # Tokenize Encoder Input
        comment_tokens = self.tokenizer(
            comment_text, 
            joined_image_roi_aspect, 
            max_length=170, 
            truncation='only_first', 
            padding='max_length', 
            return_token_type_ids=True, # Nếu dùng BERT/ViBERT thì True, XLM-R thường không quan trọng
            return_tensors='pt'
        )

        comment_input_ids = comment_tokens['input_ids'].squeeze(0)
        comment_token_type_ids = comment_tokens['token_type_ids'].squeeze(0)
        comment_attention_mask = comment_tokens['attention_mask'].squeeze(0)
        
        # Mask fusion (để mô hình biết đâu là text, đâu là ảnh feature fusion sau này)
        added_input_mask = torch.tensor([1] * (170 + 49))  # 170 token text + 49 feature ảnh

        # ----------------------------------------------------------------------
        # PHẦN 2: XỬ LÝ TARGET CHO DECODER (Quan trọng nhất)
        # ----------------------------------------------------------------------
        
        # Chuẩn hóa nhãn: "Room#Positive" -> "room positive"
        sentiment_aspects = [asp.replace("#", ' ') for asp in iaog_label]
        joined_sentiment_aspects = ' </s> '.join(sentiment_aspects)

        # Tạo chuỗi đích: <iaog> room positive </s> service negative ...
        label_str = f"<iaog> {joined_sentiment_aspects}"
        label_str = label_str.lower().replace('_', ' ')

        # Tokenize Label
        # Lưu ý: XLM-R tự động thêm <s> ở đầu và </s> ở cuối
        # Input_ids sẽ là: [<s> <iaog> tệ facilities</s> rộng public area</s><pad>, <pad>...]
        label_tokens = self.tokenizer(
            label_str, 
            max_length=self.max_len_decoder, 
            padding='max_length', 
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        full_ids = label_tokens['input_ids'].squeeze(0)
        full_mask = label_tokens['attention_mask'].squeeze(0)

        # A. Tạo DECODER INPUT IDS (Teacher Forcing Input)
        # Giữ nguyên full_ids (có <s> ở đầu) để làm gợi ý cho Decoder
        decoder_input_ids = full_ids.clone()

        # B. Tạo LABELS (Target để tính Loss)
        # Dời chuỗi sang trái 1 đơn vị: Input[t] dự đoán Label[t+1]
        labels = torch.roll(full_ids, shifts=-1, dims=0)
        
        # Xử lý phần tử cuối bị cuộn (roll) -> Mask -100
        labels[-1] = -100 
        
        # Mask tất cả các vị trí Padding thành -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask luôn vị trí EOS (</s>) trong labels nếu muốn (tùy chọn), 
        # nhưng thường ta để model dự đoán EOS để biết khi nào dừng.
        # Ở đây ta chỉ mask padding là đủ.

        decoder_attention_mask = full_mask

        # ----------------------------------------------------------------------
        # PHẦN 3: XỬ LÝ HÌNH ẢNH & ROI (Giữ nguyên logic cũ)
        # ----------------------------------------------------------------------
        list_img_features = []
        global_roi_features = []
        global_roi_coor = []

        for img_path in list_img_path[:self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            try:
                one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except:
                one_image = torch.zeros(3, 224, 224)
                img_transform = torch.zeros(1, 3, 224, 224)

            list_img_features.append(img_transform)

            # --- Crop ROI ---
            list_roi_img = [] 
            list_roi_coor = [] 
            roi_in_img_df = self.roi_df[self.roi_df['file_name'] == img_path][:self.num_roi]

            if roi_in_img_df.shape[0] == 0:
                global_roi_coor.append(np.zeros((self.num_roi, 4)))
                global_roi_features.append(np.zeros((self.num_roi, 3, 224, 224)))
                continue
            
            for i_roi in range(roi_in_img_df.shape[0]):
                x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1:5].values            
                
                # Crop và Transform ROI
                roi_in_image = one_image[:, x1:x2, y1:y2]
                if roi_in_image.numel() == 0: # Check roi rỗng
                     roi_transform = torch.zeros(3, 224, 224).numpy()
                else:
                    roi_transform = self.transform(roi_in_image).numpy()

                # Normalize toạ độ [0, 1] (Giả sử ảnh gốc 512x512 như code cũ)
                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)])
                list_roi_img.append(roi_transform)

            # Padding ROI nếu thiếu
            if i_roi < self.num_roi - 1: # -1 vì i_roi chạy từ 0
                 for k in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))

            global_roi_features.append(np.asarray(list_roi_img))
            global_roi_coor.append(np.asarray(list_roi_coor))

        # --- Padding Image Features (Batch Construction) ---
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        for i in range(min(len(list_img_features), self.num_img)):
            t_img_features[i, :] = list_img_features[i]

        # --- Padding ROI Features ---
        roi_img_features = torch.zeros((self.num_img, self.num_roi, 3, 224, 224))
        roi_coors = torch.zeros((self.num_img, self.num_roi, 4))
        
        global_roi_features = np.array(global_roi_features) 
        # Cần check dimension trước khi gán để tránh lỗi shape mismatch
        if len(global_roi_features) > 0 and len(global_roi_features.shape) > 1:
             for i in range(min(len(global_roi_features), self.num_img)):
                roi_img_features[i, :] = torch.tensor(global_roi_features[i])
                roi_coors[i, :] = torch.tensor(global_roi_coor[i])

        # --- Return actual length for valid lens ---
        source_valid_lens = torch.sum(comment_attention_mask)

        return (
            t_img_features,          # [num_img, 3, 224, 224]
            roi_img_features,        # [num_img, num_roi, 3, 224, 224]
            roi_coors,               # [num_img, num_roi, 4]
            labels,                  # [Batch, Dec_Len] (Target: Shifted & Masked -100)
            decoder_input_ids,       # [Batch, Dec_Len] (Input: Starts with <s>)
            decoder_attention_mask,  # [Batch, Dec_Len]
            comment_input_ids,       # [Batch, Enc_Len]
            comment_token_type_ids,
            comment_attention_mask,
            added_input_mask,
            source_valid_lens
        )