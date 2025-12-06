import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import os
import numpy as np

class IAOGDataset(Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=4, max_len_decoder=20):
        """
        Dataset for IAOG (Implicit Aspect-Opinion Generation) Pretraining.
        It generates data samples aligned with the 6 predefined aspects to match the main training phase.
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
        
        # Predefined 6 Aspects (Same as ViMACSA dataset)
        self.ASPECT = ['Location', 'Food', 'Room', 'Facilities', 'Service', 'Public_area']

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 1. Retrieve raw data
        idx_data = self.data.iloc[idx, :].values
        text = idx_data[0]       
        list_img_path = idx_data[1] 
        iaog_labels_raw = idx_data[6] # Expected format: ["sentiment#aspect", ...]

        # ----------------------------------------------------------------------
        # PART 1: VISUAL PROCESSING (Shared across all aspects)
        # ----------------------------------------------------------------------
        # Get visual tags (Image & ROI aspect categories)
        list_image_aspect, list_roi_aspect = self._get_visual_tags(list_img_path)
        
        # Construct the tag string for Auxiliary Sentence
        # Format: image_tags </s></s> roi_tags
        joined_tags = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
        joined_tags = joined_tags.lower().replace('_', ' ')

        # Process image pixels and ROIs coordinates
        t_img_features, roi_img_features, roi_coors = self._process_images(list_img_path)

        # ----------------------------------------------------------------------
        # PART 2: PARSE LABELS INTO DICTIONARY
        # ----------------------------------------------------------------------
        # Transform ["good#Room", "clean#Room"] -> {"Room": ["good", "clean"]}
        aspect_sentiment_map = {asp: [] for asp in self.ASPECT}
        
        if iaog_labels_raw and isinstance(iaog_labels_raw, list):
            for item in iaog_labels_raw:
                try:
                    sentiment, aspect = item.split('#')
                    # Normalize aspect name to match self.ASPECT keys
                    if aspect == "Public_area": aspect = "Public_area" 
                    
                    if aspect in aspect_sentiment_map:
                        aspect_sentiment_map[aspect].append(sentiment)
                except:
                    continue 

        # ----------------------------------------------------------------------
        # PART 3: GENERATE INPUT/OUTPUT FOR EACH ASPECT
        # ----------------------------------------------------------------------
        all_enc_inputs = {'ids': [], 'mask': [], 'type': [], 'added_mask': []}
        all_dec_inputs = {'ids': [], 'mask': [], 'labels': []}

        for aspect in self.ASPECT:
            # --- A. ENCODER INPUT ---
            # Structure: [CLS] Aspect [SEP] Text [SEP] Visual Tags [SEP]
            # Normalize aspect text for the encoder input (e.g., "Public_area" -> "public area")
            aspect_text = aspect.replace('_', ' ') 
            if "_" in aspect: aspect_text = "Public area"

            # Combine text: <Aspect> </s></s> <Review Text>
            combine_text = f"{aspect_text} </s></s> {text}" 
            combine_text = combine_text.lower().replace('_', ' ')

            # Tokenize Encoder Input
            enc_tokens = self.tokenizer(
                combine_text, 
                joined_tags, 
                max_length=170, 
                truncation='only_first', 
                padding='max_length', 
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            # Create Fusion Mask (1 for text tokens + 49 for visual tokens)
            # 170 text tokens + 49 image patches
            added_mask = torch.tensor([1] * (170 + 49))

            all_enc_inputs['ids'].append(enc_tokens['input_ids'].squeeze(0))
            all_enc_inputs['type'].append(enc_tokens['token_type_ids'].squeeze(0))
            all_enc_inputs['mask'].append(enc_tokens['attention_mask'].squeeze(0))
            all_enc_inputs['added_mask'].append(added_mask)

            # --- B. DECODER INPUT & TARGETS ---
            # Get sentiments for the current aspect loop
            sentiments = aspect_sentiment_map.get(aspect, [])
            
            if not sentiments:
                target_text = "None"
            else:
                # Sort to ensure deterministic order
                sentiments.sort()
                target_text = " , ".join(sentiments)
            
            # Decoder Input Format: <s> <iaog> sentiment1 , sentiment2 ...
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

            # Generate Labels: Shift input right by 1 position
            # Input:  <s> <iaog> good </s> <pad>
            # Target: <iaog> good </s> <pad> <pad>
            # Padding is masked with -100
            target_ids = torch.roll(dec_ids, shifts=-1, dims=0)
            target_ids[-1] = -100 # Mask the last token after rolling
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100 # Mask padding
            
            all_dec_inputs['ids'].append(dec_ids)
            all_dec_inputs['mask'].append(dec_mask)
            all_dec_inputs['labels'].append(target_ids)

        # ----------------------------------------------------------------------
        # PART 4: STACKING (Batch Dimension for Aspects)
        # ----------------------------------------------------------------------
        # Encoder Tensors: [6, Seq_Len]
        # Decoder Tensors: [6, Dec_Len]
        
        return (
            t_img_features,          # [num_img, 3, 224, 224]
            roi_img_features,        # [num_img, num_roi, 3, 224, 224]
            roi_coors,               # [num_img, num_roi, 4]
            torch.stack(all_dec_inputs['labels']),      
            torch.stack(all_dec_inputs['ids']),       
            torch.stack(all_dec_inputs['mask']),       
            torch.stack(all_enc_inputs['ids']),       
            torch.stack(all_enc_inputs['type']),        
            torch.stack(all_enc_inputs['mask']),       
            torch.stack(all_enc_inputs['added_mask']),    
            torch.sum(torch.stack(all_enc_inputs['mask']), dim=1) # Valid lengths for decoder init
        )

    def _get_visual_tags(self, list_img_path):
        """Helper to extract visual tags from JSON metadata."""
        l_img, l_roi = [], []
        for img in list_img_path[:self.num_img]:
            try: l_img.extend(self.dict_image_aspect.get(img, []))
            except: pass
            try: l_roi.extend(self.dict_roi_aspect.get(img, []))
            except: pass
            
        list_image_aspect = list(set(l_img))
        list_roi_aspect = list(set(l_roi))
        
        if len(list_image_aspect) == 0: list_image_aspect = ['empty']
        if len(list_roi_aspect) == 0: list_roi_aspect = ['empty']
        
        return list_image_aspect, list_roi_aspect

    def _process_images(self, list_img_path):
        """Helper to process image pixels and ROI coordinates."""
        # --- Logic copied from original MACSADataset to ensure compatibility ---
        list_img_features = []
        global_roi_features = []
        global_roi_coor = []

        for img_path in list_img_path[:self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            try:
                one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
                img_transform = self.transform(one_image).unsqueeze(0)
            except:
                # Handle missing images
                img_transform = torch.zeros(1, 3, 224, 224)
                one_image = torch.zeros(3, 224, 224)

            list_img_features.append(img_transform)

            # Process ROIs
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
                
                # Normalize Coordinates (0-1)
                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)])
                list_roi_img.append(roi_transform)

            # Padding ROIs if less than num_roi
            if len(list_roi_img) < self.num_roi:
                for _ in range(self.num_roi - len(list_roi_img)):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))

            global_roi_features.append(np.asarray(list_roi_img))
            global_roi_coor.append(np.asarray(list_roi_coor))

        # Padding Images if less than num_imgs
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