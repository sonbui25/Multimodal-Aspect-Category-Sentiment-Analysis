import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np

class IAOGDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=4, max_len_decoder=20):
        """
        Dataset cho bài toán IAOG (Implicit Aspect-Opinion Generation).
        Mỗi mẫu dữ liệu sẽ sinh ra 6 cặp input-output tương ứng với 6 khía cạnh cố định.
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
        
        # 6 Aspect cố định
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']

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
        text = idx_data[0]
        list_img_path = idx_data[1]
        iaog_labels_raw = idx_data[6] # Format: ["tốt#Room", "đẹp#Service", ...]

        # ----------------------------------------------------------------------
        # PHẦN 1: XỬ LÝ ẢNH (Dùng chung cho cả 6 aspect)
        # ----------------------------------------------------------------------
        list_image_aspect, list_roi_aspect = self._get_visual_tags(list_img_path)
        t_img_features, roi_img_features, roi_coors = self._process_images(list_img_path)
        
        # Tạo chuỗi tags phụ trợ: image_tags </s></s> roi_tags
        joined_tags = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
        joined_tags = joined_tags.lower().replace('_', ' ')

        # ----------------------------------------------------------------------
        # PHẦN 2: PARSE NHÃN VÀO DICTIONARY
        # ----------------------------------------------------------------------
        # Chuyển list raw thành dict: {'Room': ['tốt', 'sạch'], 'Food': []}
        aspect_sentiment_map = {asp: [] for asp in self.ASPECT}
        
        if iaog_labels_raw:
            for item in iaog_labels_raw:
                try:
                    sentiment, aspect = item.split('#')
                    if aspect == "Public_area": aspect = "Public_area"
                    
                    if aspect in aspect_sentiment_map:
                        aspect_sentiment_map[aspect].append(sentiment)
                except:
                    continue 

        # ----------------------------------------------------------------------
        # PHẦN 3: TẠO INPUT/OUTPUT CHO TỪNG ASPECT
        # ----------------------------------------------------------------------
        all_enc_inputs = {'ids': [], 'mask': [], 'type': [], 'added_mask': []}
        all_dec_inputs = {'ids': [], 'mask': [], 'labels': []}

        for aspect in self.ASPECT:
            # --- A. XỬ LÝ NHÃN (DECODER TARGET) ---
            sentiments = aspect_sentiment_map.get(aspect, [])
            
            if not sentiments:
                target_str = "None"
            else:
                # Sắp xếp để đảm bảo thứ tự nhất quán
                target_str = " , ".join(sorted(sentiments))
            
            # --- B. ENCODER INPUT ---
            # Format: <Aspect> </s></s> <Text> </s></s> <Visual Tags>
            asp_text = "Public area" if aspect == "Public_area" else aspect
            combine_text = f"{asp_text} </s></s> {text}".lower().replace('_', ' ')
            
            enc = self.tokenizer(
                combine_text, 
                joined_tags, 
                max_length=170, 
                truncation='only_first', 
                padding='max_length', 
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            # Mask cho fusion (170 text + 49 image patches)
            added_mask = torch.tensor([1] * (170 + 49))

            all_enc_inputs['ids'].append(enc['input_ids'].squeeze(0))
            all_enc_inputs['mask'].append(enc['attention_mask'].squeeze(0))
            all_enc_inputs['type'].append(enc['token_type_ids'].squeeze(0))
            all_enc_inputs['added_mask'].append(added_mask)

            # --- C. DECODER INPUT ---
            # Format: <s> <iaog> sentiment1 , sentiment2 ...
            dec_text = f"<iaog> {target_str}".lower()
            dec = self.tokenizer(
                dec_text,
                max_length=self.max_len_decoder,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            d_ids = dec['input_ids'].squeeze(0)
            d_mask = dec['attention_mask'].squeeze(0)
            
            # Tạo Label: Shift input sang phải 1 token (Standard Seq2Seq)
            # Input:  <s> <iaog> tốt </s> <pad>
            # Target: <iaog> tốt </s> <pad> <pad> (-100 tại padding)
            lbls = torch.roll(d_ids, shifts=-1, dims=0)
            lbls[-1] = -100
            lbls[lbls == self.tokenizer.pad_token_id] = -100
            
            all_dec_inputs['ids'].append(d_ids)
            all_dec_inputs['mask'].append(d_mask)
            all_dec_inputs['labels'].append(lbls)

        # ----------------------------------------------------------------------
        # PHẦN 4: TRẢ VỀ TENSORS
        # ----------------------------------------------------------------------
        # Stack lại để tạo batch dimension cho aspect: [6, Seq_Len]
        
        return (
            t_img_features,          # [num_img, 3, 224, 224]
            roi_img_features,        # [num_img, num_roi, 3, 224, 224]
            roi_coors,               # [num_img, num_roi, 4]
            torch.stack(all_dec_inputs['labels']),      # [6, Dec_Len]
            torch.stack(all_dec_inputs['ids']),         # [6, Dec_Len]
            torch.stack(all_dec_inputs['mask']),        # [6, Dec_Len]
            torch.stack(all_enc_inputs['ids']),         # [6, Enc_Len]
            torch.stack(all_enc_inputs['type']),        # [6, Enc_Len]
            torch.stack(all_enc_inputs['mask']),        # [6, Enc_Len]
            torch.stack(all_enc_inputs['added_mask']),  # [6, 170+49]
            torch.sum(torch.stack(all_enc_inputs['mask']), dim=1), # Valid lengths [6]
            text # Trả về text gốc để phục vụ Logging
        )

    def _get_visual_tags(self, list_img_path):
        """Hàm phụ trợ lấy visual tags"""
        l_img, l_roi = [], []
        for img in list_img_path[:self.num_img]:
            try: l_img.extend(self.dict_image_aspect.get(img, []))
            except: pass
            try: l_roi.extend(self.dict_roi_aspect.get(img, []))
            except: pass
        return list(set(l_img) or ['empty']), list(set(l_roi) or ['empty'])

    def _process_images(self, list_img_path):
        """Hàm phụ trợ xử lý ma trận ảnh và ROI"""
        # (Logic giữ nguyên từ các phiên bản trước)
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
                one_image = torch.zeros(3, 224, 224)

            list_img_features.append(img_transform)
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
                if roi_in_image.numel() == 0: roi_transform = torch.zeros(3, 224, 224).numpy()
                else: roi_transform = self.transform(roi_in_image).numpy()
                
                # Normalize Coordinates
                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)])
                list_roi_img.append(roi_transform)

            if len(list_roi_img) < self.num_roi:
                 for k in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))
            global_roi_features.append(np.asarray(list_roi_img))
            global_roi_coor.append(np.asarray(list_roi_coor))

        t_img = torch.zeros(self.num_img, 3, 224, 224)
        for i in range(min(len(list_img_features), self.num_img)):
            t_img[i, :] = list_img_features[i]
        
        roi_img = torch.zeros(self.num_img, self.num_roi, 3, 224, 224)
        roi_coor = torch.zeros(self.num_img, self.num_roi, 4)
        
        global_roi_features = np.array(global_roi_features) 
        if len(global_roi_features) > 0 and len(global_roi_features.shape) > 1:
             for i in range(min(len(global_roi_features), self.num_img)):
                roi_img[i, :] = torch.tensor(global_roi_features[i])
                roi_coor[i, :] = torch.tensor(global_roi_coor[i])
        
        return t_img, roi_img, roi_coor