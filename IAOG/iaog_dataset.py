from torchvision.io import read_image, ImageReadMode
import torch
from torchvision import transforms
from transformers import AutoTokenizer
import os 
import numpy as np

# by default: num_img = num_roi = 7
# return:
#   [CLS] text [SEP] list image aspect (split by , ) [SEP] list roi aspect (split by ,) [SEP]
#   image: (num_img, 3, 224, 224)
#   roi: (num_img, num_roi, 3, 224, 224) 
from torchvision.io import read_image, ImageReadMode
import torch
from torchvision import transforms
from transformers import AutoTokenizer
import os
import numpy as np

class IAOGDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, img_folder, roi_df, dict_image_aspect, dict_roi_aspect, num_img=7, num_roi=7):
        self.data = data
        self.tokenizer = tokenizer
        self.img_folder = img_folder
        self.roi_df = roi_df
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi

        # --- Transform Image ---
        self.transform = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        idx_data = self.data.iloc[idx, :].values
        text = idx_data[0]
        list_img_path = idx_data[1]

        # --- Process Text Aspects ---
        list_image_aspect = []
        list_roi_aspect = []
        for img_name in list_img_path[:self.num_img]:
            try: list_image_aspect.extend(self.dict_image_aspect[img_name])
            except: pass
            try: list_roi_aspect.extend(self.dict_roi_aspect[img_name])
            except: pass

        list_image_aspect = list(set(list_image_aspect))
        list_roi_aspect = list(set(list_roi_aspect))

        if len(list_image_aspect) == 0: list_image_aspect = ['empty']
        if len(list_roi_aspect) == 0: list_roi_aspect = ['empty']

        # --- Process IAOG Label ---
        iaog_label = idx_data[6]
        sentiment_aspects = []
        for sentiment_asp in iaog_label:
            sentiment_asp = sentiment_asp.replace("#", ' ')
            sentiment_aspects.append(sentiment_asp)

        joined_sentiment_aspects = ' </s> '.join(sentiment_aspects)
        joined_image_roi_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
        joined_image_roi_aspect = joined_image_roi_aspect.lower().replace('_',' ')

        # --- Decoder Input ---
        iaog_text = f"<iaog> {joined_sentiment_aspects}"
        iaog_text = iaog_text.lower().replace('_',' ')
        iaog_tokens = self.tokenizer(iaog_text, max_length=20, padding='max_length', truncation=True)

        iaog_input_ids = torch.tensor(iaog_tokens['input_ids'])
        iaog_attention_mask = torch.tensor(iaog_tokens['attention_mask'])

        # --- Encoder Input ---
        comment_text = f"{text}"
        comment_text = comment_text.lower().replace('_',' ')
        comment_tokens = self.tokenizer(comment_text, joined_image_roi_aspect, max_length=170, truncation='only_first', padding='max_length', return_token_type_ids=True)

        comment_input_ids = torch.tensor(comment_tokens['input_ids'])
        comment_token_type_ids = torch.tensor(comment_tokens['token_type_ids'])
        comment_attention_mask = torch.tensor(comment_tokens['attention_mask'])
        added_attention_mask = torch.tensor([1] * (170 + 49 + self.num_roi))

        # --- Process Image & ROI ---
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
                list_roi_img = np.zeros((self.num_roi, 3, 224, 224))
                global_roi_coor.append(np.zeros((self.num_roi, 4)))
                global_roi_features.append(list_roi_img)
                continue

            count_roi = 0
            for i_roi in range(roi_in_img_df.shape[0]):
                x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1:5].values
                roi_in_image = one_image[:, int(x1):int(x2), int(y1):int(y2)]

                if roi_in_image.shape[1] == 0 or roi_in_image.shape[2] == 0:
                     roi_transform = np.zeros((3, 224, 224))
                else:
                     roi_transform = self.transform(roi_in_image).numpy()

                x1, x2, y1, y2 = x1/512, x2/512, y1/512, y2/512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]

                list_roi_coor.append([cv(x1), cv(x2), cv(y1), cv(y2)])
                list_roi_img.append(roi_transform)
                count_roi += 1

            # --- Padding ROI ---
            if count_roi < self.num_roi:
                for k in range(self.num_roi - count_roi):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))

            global_roi_features.append(list_roi_img)
            global_roi_coor.append(list_roi_coor)

        # --- Padding Image ---
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        num_imgs = len(list_img_features)
        for i in range(self.num_img):
            if i < num_imgs:
                t_img_features[i, :] = list_img_features[i]

        # --- Padding Global ROI ---
        global_roi_features = np.asarray(global_roi_features)
        global_roi_coor = np.asarray(global_roi_coor)

        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        roi_coors = np.zeros((self.num_img, self.num_roi, 4))

        num_img_roi = len(global_roi_features)
        for i in range(self.num_img):
            if i < num_img_roi:
                roi_img_features[i, :] = global_roi_features[i]
                roi_coors[i, :] = global_roi_coor[i]

        roi_img_features = torch.tensor(roi_img_features).float()
        roi_coors = torch.tensor(roi_coors).float()

        return (
            t_img_features,
            roi_img_features,
            roi_coors,
            iaog_input_ids,
            iaog_attention_mask,
            comment_input_ids,
            comment_token_type_ids,
            comment_attention_mask,
            added_attention_mask
        )