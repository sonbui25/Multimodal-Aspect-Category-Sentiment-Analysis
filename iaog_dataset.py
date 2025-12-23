import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np
import pandas as pd

class IAOGDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=4, max_len_decoder=20):
        self.data = data
        self.tokenizer = tokenizer
        self.img_folder = img_folder
        self.roi_df = roi_df
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi
        self.max_len_decoder = max_len_decoder
        
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']
        self.aspect2id = {a: i for i, a in enumerate(self.ASPECT)}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # --- LOGIC MỚI: FLATTEN DỮ LIỆU ---
        self.samples = []
        for idx, row in self.data.iterrows():
            iaog_labels_raw = row.get('iaog_labels', [])
            
            if not isinstance(iaog_labels_raw, list) or len(iaog_labels_raw) == 0:
                continue

            # 1. Gom nhóm từ cảm xúc theo Aspect
            aspect_group = {}
            for label_str in iaog_labels_raw:
                if '#' not in label_str: continue
                parts = label_str.split('#')
                sentiment_word = parts[0].strip()
                aspect_name = parts[1].strip()
                if aspect_name == "Public_area": aspect_name = "Public_area"

                if aspect_name in self.aspect2id:
                    if aspect_name not in aspect_group: aspect_group[aspect_name] = []
                    if sentiment_word not in aspect_group[aspect_name]: aspect_group[aspect_name].append(sentiment_word)
            
            # 2. Tạo mẫu training riêng biệt cho từng Aspect
            for aspect, words in aspect_group.items():
                target_string = " , ".join(sorted(words))
                self.samples.append({
                    'original_idx': idx,
                    'target_aspect': aspect,
                    'target_sentiment': target_string
                })
        
        print(f"--> IAOG Dataset Loaded (Positive-Only). Total Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        idx_data = self.data.iloc[sample['original_idx']]
        text = idx_data['comment']
        list_img_path = idx_data['list_img']
        
        target_aspect = sample['target_aspect']
        target_sentiment = sample['target_sentiment']

        # 1. VISUAL
        list_image_aspect, list_roi_aspect = self._get_visual_tags(list_img_path)
        t_img_features, roi_img_features, roi_coors = self._process_images(list_img_path)
        joined_tags = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}".lower().replace('_', ' ')

        # 2. ENCODER INPUT
        asp_text = "Public area" if target_aspect == "Public_area" else target_aspect
        combine_text = f"{asp_text} </s></s> {text}".lower().replace('_', ' ')
        
        enc = self.tokenizer(combine_text, joined_tags, max_length=170, truncation='only_first', padding='max_length', return_token_type_ids=True, return_tensors='pt')
        enc_ids = enc['input_ids'].squeeze(0)
        enc_mask = enc['attention_mask'].squeeze(0)
        enc_type = enc['token_type_ids'].squeeze(0)
        added_mask = torch.tensor([1] * (170 + 49))

        # 3. DECODER
        dec_text = f"<iaog> {target_sentiment}".lower()
        dec = self.tokenizer(dec_text, max_length=self.max_len_decoder, padding='max_length', truncation=True, return_tensors='pt')
        dec_input_ids = dec['input_ids'].squeeze(0)
        labels = torch.roll(dec_input_ids, shifts=-1, dims=0)
        labels[-1] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # TRẢ VỀ 11 PHẦN TỬ
        return (
            t_img_features, roi_img_features, roi_coors, 
            labels, dec_input_ids, enc_ids, enc_type, enc_mask, added_mask,
            target_aspect, text
        )

    def _get_visual_tags(self, list_img_path):
        l_img, l_roi = [], []
        if list_img_path is None: list_img_path = []
        for img in list_img_path[:self.num_img]:
            try: l_img.extend(self.dict_image_aspect.get(img, [])); l_roi.extend(self.dict_roi_aspect.get(img, []))
            except: pass
        return list(set(l_img) or ['empty']), list(set(l_roi) or ['empty'])

    def _process_images(self, list_img_path):
        list_img_features, global_roi_features, global_roi_coor = [], [], []
        if list_img_path is None: list_img_path = []
        for img_path in list_img_path[:self.num_img]:
            try:
                one_image = read_image(os.path.join(self.img_folder, img_path), mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except: img_transform = torch.zeros(1, 3, 224, 224); one_image = torch.zeros(3, 224, 224)
            list_img_features.append(img_transform)
            
            list_roi_img, list_roi_coor = [], []
            try: roi_in_img_df = self.roi_df[self.roi_df['file_name'] == img_path][:self.num_roi]
            except: roi_in_img_df = pd.DataFrame()
            
            if roi_in_img_df.shape[0] == 0:
                global_roi_coor.append(np.zeros((self.num_roi, 4))); global_roi_features.append(np.zeros((self.num_roi, 3, 224, 224)))
                continue

            for i_roi in range(roi_in_img_df.shape[0]):
                x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1:5].values
                max_h, max_w = one_image.shape[1], one_image.shape[2]
                x1, x2 = max(0, int(x1)), min(max_h, int(x2)); y1, y2 = max(0, int(y1)), min(max_w, int(y2))
                roi_in_image = one_image[:, x1:x2, y1:y2]
                roi_transform = self.transform(roi_in_image).numpy() if roi_in_image.numel() > 0 else torch.zeros(3, 224, 224).numpy()
                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)]); list_roi_img.append(roi_transform)

            if len(list_roi_img) < self.num_roi:
                 for k in range(self.num_roi - len(list_roi_img)): list_roi_img.append(np.zeros((3, 224, 224))); list_roi_coor.append(np.zeros((4,)))
            global_roi_features.append(np.asarray(list_roi_img)); global_roi_coor.append(np.asarray(list_roi_coor))

        t_img = torch.zeros(self.num_img, 3, 224, 224)
        for i in range(min(len(list_img_features), self.num_img)): t_img[i, :] = list_img_features[i]
        roi_img = torch.zeros(self.num_img, self.num_roi, 3, 224, 224)
        roi_coor = torch.zeros(self.num_img, self.num_roi, 4)
        global_roi_features = np.array(global_roi_features) 
        if len(global_roi_features) > 0 and len(global_roi_features.shape) > 1:
             for i in range(min(len(global_roi_features), self.num_img)):
                roi_img[i, :] = torch.tensor(global_roi_features[i]); roi_coor[i, :] = torch.tensor(global_roi_coor[i])
        return t_img, roi_img, roi_coor