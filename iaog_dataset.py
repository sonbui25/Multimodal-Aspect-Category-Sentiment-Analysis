import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np

class IAOGDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=4, max_len_decoder=16):
        """
        Dataset cho bài toán IAOG (Implicit Aspect-Opinion Generation) theo phong cách Aspect-based.
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
        
        # Định nghĩa 6 Aspect cố định để lặp (Giống vimacsa_dataset)
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']

        # Transform ảnh
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
        list_img_path = idx_data[1] # List đường dẫn ảnh
        iaog_labels_raw = idx_data[6] # List labels: ["sạch sẽ#Public_area", ...]

        # ----------------------------------------------------------------------
        # PHẦN 1: XỬ LÝ ẢNH & ROI (Dùng chung cho cả mẫu, không phụ thuộc Aspect)
        # ----------------------------------------------------------------------
        list_image_aspect = []
        list_roi_aspect = []
        
        # ... (Logic lấy tag ảnh/roi giống hệt vimacsa_dataset) ...
        for img_name in list_img_path[:self.num_img]:
            try: list_image_aspect.extend(self.dict_image_aspect[img_name])
            except: pass
            try: list_roi_aspect.extend(self.dict_roi_aspect[img_name])
            except: pass

        list_image_aspect = list(set(list_image_aspect))
        list_roi_aspect = list(set(list_roi_aspect))
        if len(list_image_aspect) == 0: list_image_aspect = ['empty']
        if len(list_roi_aspect) == 0: list_roi_aspect = ['empty']

        # Chuỗi tag phụ trợ
        joined_tags = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
        joined_tags = joined_tags.lower().replace('_', ' ')

        # Xử lý ma trận ảnh (Pixel values)
        t_img_features, roi_img_features, roi_coors = self._process_images(list_img_path)

        # ----------------------------------------------------------------------
        # PHẦN 2: PARSE LABEL VÀO DICTIONARY
        # ----------------------------------------------------------------------
        # Chuyển ["tốt#Room", "đẹp#Room"] -> {"Room": ["tốt", "đẹp"]}
        aspect_sentiment_map = {asp: [] for asp in self.ASPECT}
        
        if iaog_labels_raw and len(iaog_labels_raw) > 0:
            for item in iaog_labels_raw:
                try:
                    sentiment, aspect = item.split('#')
                    # Map aspect name nếu cần (ví dụ Public_area -> Public area) để khớp key
                    if aspect == "Public_area": aspect = "Public_area" # Giữ nguyên hoặc map tùy data gốc
                    
                    if aspect in aspect_sentiment_map:
                        aspect_sentiment_map[aspect].append(sentiment)
                except:
                    continue # Bỏ qua lỗi format

        # ----------------------------------------------------------------------
        # PHẦN 3: LOOP QUA TỪNG ASPECT ĐỂ TẠO INPUT/OUTPUT (Cốt lõi)
        # ----------------------------------------------------------------------
        all_enc_input_ids = []
        all_enc_attn_mask = []
        all_enc_token_type_ids = []
        all_added_input_mask = []
        
        all_dec_input_ids = []
        all_dec_attn_mask = []
        all_labels = []

        for aspect in self.ASPECT:
            # --- A. ENCODER INPUT (Format: Aspect </s> Text </s> Tags) ---
            # Xử lý tên aspect cho khớp tokenizer (thường lower case và bỏ underscore)
            aspect_text = aspect.replace('_', ' ') 
            if "_" in aspect: aspect_text = "Public area" # Chuẩn hóa giống vimacsa_dataset

            # [QUAN TRỌNG] Cấu trúc giống hệt giai đoạn training chính
            # encoder input: <aspect> </s></s> <text> </s></s> <tags>
            combine_text = f"{aspect_text} </s></s> {text}" 
            combine_text = combine_text.lower().replace('_', ' ')

            enc_tokens = self.tokenizer(
                combine_text, 
                joined_tags, 
                max_length=170, 
                truncation='only_first', 
                padding='max_length', 
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            # Mask fusion (text + visual tokens placeholders)
            added_mask = torch.tensor([1] * (170 + 49))

            all_enc_input_ids.append(enc_tokens['input_ids'].squeeze(0))
            all_enc_token_type_ids.append(enc_tokens['token_type_ids'].squeeze(0))
            all_enc_attn_mask.append(enc_tokens['attention_mask'].squeeze(0))
            all_added_input_mask.append(added_mask)

            # --- B. DECODER INPUT & LABELS ---
            # Lấy list sentiment tương ứng aspect này
            sentiments = aspect_sentiment_map.get(aspect, [])
            if not sentiments:
                target_text = "None"
            else:
                # Sắp xếp để nhất quán (tránh lúc A,B lúc B,A)
                sentiments.sort()
                target_text = " , ".join(sentiments)
            
            # Format: <iaog> sentiment1 , sentiment2
            decoder_text = f"<iaog> {target_text}".lower()

            dec_tokens = self.tokenizer(
                decoder_text,
                max_length=self.max_len_decoder,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            dec_ids = dec_tokens['input_ids'].squeeze(0)
            dec_mask = dec_tokens['attention_mask'].squeeze(0)

            # Tạo Label: Shift phải input 1 token và mask padding
            # Input:  <s> <iaog> sent </s> <pad>
            # Target: <iaog> sent </s> <pad> <pad>
            target_ids = dec_ids.clone()
            
            # Logic shift labels (dựa trên pretraining chuẩn):
            # Target tại t là dự đoán từ input 0...t.
            # Thông thường thư viện HF tự shift bên trong model nếu truyền labels.
            # Nhưng nếu code loss tự viết, ta chuẩn bị labels:
            # Labels = Input dịch sang trái 1 bước (bỏ token đầu tiên - thường là <s>)
            # Input Decoder: <s> <iaog> None </s>
            # Output mong muốn: <iaog> None </s>
            
            # Tuy nhiên, cách an toàn nhất cho custom seq2seq:
            # Dec Input: [Start] A B C
            # Label:     A B C [End]
            
            # Ở đây ta dùng logic cuộn (roll) như code cũ, nhưng mask cẩn thận:
            lbls = torch.roll(dec_ids, shifts=-1, dims=0)
            # Mask token cuối cùng sau khi roll (vì nó bị đẩy từ đầu xuống)
            lbls[-1] = -100
            # Mask padding
            lbls[lbls == self.tokenizer.pad_token_id] = -100
            
            all_dec_input_ids.append(dec_ids)
            all_dec_attn_mask.append(dec_mask)
            all_labels.append(lbls)

        # ----------------------------------------------------------------------
        # PHẦN 4: STACKING (Tạo batch dimension cho Aspect)
        # ----------------------------------------------------------------------
        # Kích thước: [6, 170] (6 là số aspect)
        all_enc_input_ids = torch.stack(all_enc_input_ids)
        all_enc_token_type_ids = torch.stack(all_enc_token_type_ids)
        all_enc_attn_mask = torch.stack(all_enc_attn_mask)
        all_added_input_mask = torch.stack(all_added_input_mask)
        
        # Kích thước: [6, max_len_decoder]
        all_dec_input_ids = torch.stack(all_dec_input_ids)
        all_dec_attn_mask = torch.stack(all_dec_attn_mask)
        all_labels = torch.stack(all_labels)

        # Valid lens cho encoder (để init decoder state, nếu cần)
        # Lấy sum mask của aspect đầu tiên (hoặc trung bình, nhưng text input độ dài khác nhau do tên aspect)
        # Ở đây trả về tensor [6] chứa độ dài thật của từng câu input theo aspect
        source_valid_lens = torch.sum(all_enc_attn_mask, dim=1)
        # In decoder input IDs và encoder input IDs và labels của toàn bộ aspect để debug
        for i in range(len(self.ASPECT)):
            print(f"Decoder Input IDs for Aspect '{self.ASPECT[i]}':", self.tokenizer.decode(all_dec_input_ids[i]))
            print(f"Encoder Input IDs for Aspect '{self.ASPECT[i]}':", self.tokenizer.decode(all_enc_input_ids[i]))
            print(f"Labels for Aspect '{self.ASPECT[i]}':", self.tokenizer.decode(all_labels[i][all_labels[i] != -100]))
            print("\n")
        
        return (
            t_img_features,          # [num_img, 3, 224, 224] (Chung cho cả 6 aspect)
            roi_img_features,        # [num_img, num_roi, 3, 224, 224] (Chung)
            roi_coors,               # [num_img, num_roi, 4] (Chung)
            all_labels,              # [6, Dec_Len]
            all_dec_input_ids,       # [6, Dec_Len]
            all_dec_attn_mask,       # [6, Dec_Len]
            all_enc_input_ids,       # [6, Enc_Len]
            all_enc_token_type_ids,  # [6, Enc_Len]
            all_enc_attn_mask,       # [6, Enc_Len]
            all_added_input_mask,    # [6, 170+49]
            source_valid_lens        # [6]
        )

    def _process_images(self, list_img_path):
        """Hàm phụ trợ xử lý ảnh để code gọn hơn"""
        list_img_features = []
        global_roi_features = []
        global_roi_coor = []

        for img_path in list_img_path[:self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            try:
                one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except:
                img_transform = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224) # Placeholder

            list_img_features.append(img_transform)

            # --- ROI Processing ---
            list_roi_img = []
            list_roi_coor = []
            roi_in_img_df = self.roi_df[self.roi_df['file_name'] == img_path][:self.num_roi]

            if roi_in_img_df.shape[0] == 0:
                global_roi_coor.append(np.zeros((self.num_roi, 4)))
                global_roi_features.append(np.zeros((self.num_roi, 3, 224, 224)))
                continue

            for i_roi in range(roi_in_img_df.shape[0]):
                x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1:5].values
                roi_in_image = one_image[:, x1:x2, y1:y2]
                if roi_in_image.numel() == 0:
                    roi_transform = torch.zeros(3, 224, 224).numpy()
                else:
                    roi_transform = self.transform(roi_in_image).numpy()
                
                # Normalize Coordinates
                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)])
                list_roi_img.append(roi_transform)

            # Padding ROIs
            if len(list_roi_img) < self.num_roi:
                for _ in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))

            global_roi_features.append(np.asarray(list_roi_img))
            global_roi_coor.append(np.asarray(list_roi_coor))

        # Padding Images (Sequence level)
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        for i in range(min(len(list_img_features), self.num_img)):
            t_img_features[i, :] = list_img_features[i]

        roi_img_features = torch.zeros((self.num_img, self.num_roi, 3, 224, 224))
        roi_coors = torch.zeros((self.num_img, self.num_roi, 4))
        
        global_roi_features = np.array(global_roi_features)
        if len(global_roi_features) > 0 and len(global_roi_features.shape) > 1:
            for i in range(min(len(global_roi_features), self.num_img)):
                roi_img_features[i, :] = torch.tensor(global_roi_features[i])
                roi_coors[i, :] = torch.tensor(global_roi_coor[i])

        return t_img_features, roi_img_features, roi_coors