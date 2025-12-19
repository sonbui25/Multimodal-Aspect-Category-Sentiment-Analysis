import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModel
import re
import torch
import os
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import argparse
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm,trange
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import random
import logging
import sys

# --- MODIFIED: Thêm filename vào return của Dataset ---
class RoiDataset(Dataset):
    def __init__(self, data, root_dir, ASPECT):
        self.image_label = data
        self.root_dir = root_dir
        self.ASPECT = ASPECT
    def __len__(self):
        return self.image_label.shape[0]
    def __getitem__(self,index):
        image_name = self.image_label.loc[index, "file_name"] + ".png"
        x1, x2, y1, y2 = self.image_label.iloc[index,1:4+1].values

        image = torchvision.io.read_image(os.path.join(self.root_dir,image_name),mode = torchvision.io.ImageReadMode.RGB) 
        image = image[:,x1:x2,y1:y2]

        text_lb = self.image_label.loc[index,'label']
        num_lb = self.ASPECT.index(text_lb)        

        transforms = v2.Compose([
                            v2.Resize((224,224),antialias=True),
                            v2.RandomHorizontalFlip(),
                            v2.ConvertImageDtype(torch.float32),
                            v2.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))
                                        ])

        image = transforms(image)

        return {
            "image": image,
            "label": num_lb,
            "filename": image_name # <--- Added this
        }
        
class MyRoIModel(torch.nn.Module):
  def __init__(self,num_classes):
    super(MyRoIModel, self).__init__()
    self.feature_extractor = torchvision.models.resnet152(weights = ResNet152_Weights.IMAGENET1K_V2)
    self.no_fc = torch.nn.Sequential(*(list(self.feature_extractor.children())[:-1]))
    self.linear = torch.nn.Linear(2048,num_classes)
  def forward(self, input):
    img_features = self.no_fc(input).squeeze(-1).squeeze(-1)
    out = self.linear(img_features)
    return out

def load_model(path):
    check_point = torch.load(path,map_location=torch.device('cpu'))
    return check_point

def save_model(path, model, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    },path)

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, _ \
      = precision_recall_fscore_support(y_true, y_pred,zero_division=0.0,labels = [0,1,2,3,4])
    return p_macro, r_macro, f_macro

def convert_img_to_tensor(img):
    transforms = v2.Compose([
                        v2.Resize((224,224),antialias=True),
                        v2.ConvertImageDtype(torch.float32),
                        v2.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                                    ])
    image = transforms(img)
    return image
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../image', type=str, required=True)
    parser.add_argument("--roi_label_path", default=None, type=str, required=True)
    parser.add_argument("--weight_path", default=None, type=str)
    parser.add_argument("--output_dir", default="../vimacsa", type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--get_cate", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=8.0, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')
    
    args = parser.parse_args()
    print("===================== RUN ROI CATEGORIES =====================")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(
        format=log_format,
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{args.output_dir}/roi_categories.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    if args.no_cuda: device = 'cpu'
    else: device = 'cuda'

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if not args.do_train and not args.get_cate:
        raise ValueError("At least one of `do_train` or `get_cate` must be True.")
    
    ASPECT = ['food', 'room', 'facilities', 'service', 'public_area'] 

    if args.do_train:
        if args.roi_label_path == None: raise ValueError("Please provide annotated RoI file.")

        roi_df = pd.read_csv(f"{args.roi_label_path}")
        train_data, dev_test_data = train_test_split(roi_df,test_size=0.3,random_state=18)
        dev_data, test_data = train_test_split(dev_test_data,test_size=0.5,random_state=18)

        train_data = train_data.reset_index().drop('index',axis=1)
        dev_data = dev_data.reset_index().drop('index',axis=1)
        test_data = test_data.reset_index().drop('index',axis=1)

        train_set = RoiDataset(train_data,args.image_dir,ASPECT)
        dev_set = RoiDataset(dev_data,args.image_dir,ASPECT)
        test_set = RoiDataset(test_data,args.image_dir,ASPECT)

        train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle = True)
        dev_loader = DataLoader(dev_set,batch_size=args.eval_batch_size,shuffle = False)
        test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle = False)

        num_train_steps = len(train_loader)*args.num_train_epochs

        model = MyRoIModel(len(ASPECT)) 
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
        max_accracy = 0.0

        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            # ... Training loop ...
            model.train()
            for step, batch in enumerate(tqdm(train_loader, position=0, leave=False, desc="Train")):
                input = batch['image'].to(device)
                label = batch['label'].to(device)
                logits = model(input)
                loss = criterion(logits,label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logger.info("***** Running evaluation on Dev Set*****")
            model.eval()
            idx2asp = {i:v for i,v in enumerate(ASPECT)}
            eval_epoch_loss = 0
            all_truth = []
            all_pred = []
            
            for step, batch in enumerate(tqdm(dev_loader, position=0, leave=True, desc="Dev")):
                input = batch['image'].to(device)
                label = batch['label'].to(device)
                with torch.no_grad():
                    logits = model(input)
                    loss = criterion(logits,label)
                    eval_epoch_loss += loss.item()
                    logits = logits.cpu().numpy()
                    pred = np.argmax(logits, axis = -1)
                    all_pred.extend(pred.tolist())
                    all_truth.extend(label.cpu().numpy().tolist())

            if step == 0: step = 1
            eval_epoch_loss /= step 
            
            all_precision, all_recall, all_f1 = macro_f1(all_truth, all_pred)
            all_f1_mean = all_f1.mean()
            matrix = confusion_matrix(all_truth, all_pred,labels = [i for i in range(len(ASPECT))])
            all_accuracy = matrix.diagonal()/matrix.sum(axis=1)
            all_accuracy = np.where(np.isnan(all_accuracy), 0, all_accuracy).mean()

            logger.info(f"Epoch {train_idx} Dev Results - F1: {all_f1_mean:.4f}, Acc: {all_accuracy:.4f}")

            if all_accuracy >= max_accracy:   
                save_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth',model,train_idx)
                max_accracy = all_accuracy
                logger.info(f"New Best Accuracy: {max_accracy:.4f}")

        # --- TEST SECTION (Modified) ---
        output_test_file = os.path.join(args.output_dir, "test_roi_results.txt")
        output_detail_file = os.path.join(args.output_dir, "test_roi_predictions_detail.txt")
        
        logger.info("***** Running evaluation on Test Set *****")
        checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_all_truth = []
        test_all_pred = []
        detailed_logs = []

        for step, batch in enumerate(tqdm(test_loader, position=0, leave=True, desc="Test")):
            input = batch['image'].to(device)
            label = batch['label'].to(device)
            filenames = batch['filename'] # Lấy tên file

            with torch.no_grad():
                logits = model(input)
                logits_np = logits.cpu().numpy()
                pred_batch = np.argmax(logits_np, axis = -1)
                
                label_np = label.cpu().numpy()

                test_all_pred.extend(pred_batch.tolist())
                test_all_truth.extend(label_np.tolist())

                # --- Xử lý Log chi tiết ---
                batch_size = input.shape[0]
                for i in range(batch_size):
                    fname = filenames[i]
                    # ROI chỉ có 1 nhãn duy nhất (Single Label Classification)
                    # Nhưng để đúng format bạn yêu cầu (danh sách), ta bọc nó vào list []
                    pred_name = ASPECT[pred_batch[i]]
                    gold_name = ASPECT[label_np[i]]
                    
                    # Sort list (chỉ có 1 phần tử nên ko thay đổi gì, nhưng giữ logic code)
                    pred_list = sorted([pred_name])
                    gold_list = sorted([gold_name])
                    
                    detailed_logs.append({
                        "file": fname,
                        "gold": gold_list,
                        "pred": pred_list
                    })

        # --- Ghi metrics ---
        with open(output_test_file, "w") as writer:
            writer.write("***** TEST RESULTS (ROI Categories) *****\n")
            test_all_precision, test_all_recall, test_all_f1 = macro_f1(test_all_truth, test_all_pred)
            matrix = confusion_matrix(test_all_truth, test_all_pred,labels = [i for i in range(len(ASPECT))])
            test_all_accuracy = matrix.diagonal()/matrix.sum(axis=1)
            test_all_accuracy = np.where(np.isnan(test_all_accuracy), 0, test_all_accuracy)

            for id_asp in range(len(ASPECT)):
                logger.info(f"Aspect: {idx2asp[id_asp]:<15} | F1: {test_all_f1[id_asp]:.4f} | Acc: {test_all_accuracy[id_asp]:.4f}")
                writer.write(f"{idx2asp[id_asp]:<20} | F1: {test_all_f1[id_asp]:.4f} | Acc: {test_all_accuracy[id_asp]:.4f}\n")
        
        # --- Ghi log chi tiết theo format ---
        with open(output_detail_file, "w", encoding='utf-8') as f:
            for item in detailed_logs:
                f.write(f'"{item["file"]}": [\n')
                f.write(f'    Gold_Label: {json.dumps(item["gold"], ensure_ascii=False)},\n')
                f.write(f'    Prediction: {json.dumps(item["pred"], ensure_ascii=False)},\n')
                f.write('  ],\n')

        logger.info(f"Saved detailed formatted predictions to {output_detail_file}")

    if args.get_cate:
        print("===================== GET ROI CATEGORIES =====================")
        model = MyRoIModel(len(ASPECT))
        if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
        model = model.to(device)

        if args.do_train: checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_roi_model.pth')
        else: checkpoint = load_model(args.weight_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        ASPECT = np.asarray(ASPECT)
        image_label_dict = {}

        roi_df = pd.read_csv(f"{args.roi_label_path}")
        list_img_name = roi_df['file_name'].unique()
        
        for img_name in tqdm(list_img_name, desc="Inferencing"):
            p = img_name + ".png"
            image = torchvision.io.read_image(os.path.join(args.image_dir,p),mode = torchvision.io.ImageReadMode.RGB)
            df_roi = roi_df[roi_df['file_name']==img_name][:6].reset_index()
            num_roi = df_roi.shape[0]

            image_aspect = []
            for i in range(num_roi):
                x1 = df_roi.loc[i,'x1']; x2 = df_roi.loc[i,'x2']
                y1 = df_roi.loc[i,'y1']; y2 = df_roi.loc[i,'y2']
                roi_img = image[:,x1:x2,y1:y2]
                roi_img = convert_img_to_tensor(roi_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(roi_img)
                    pred = np.argmax(pred.cpu().numpy(),axis=-1)
                    image_aspect.append(ASPECT[pred][0])
            
            image_aspect = list(set(image_aspect))
            image_label_dict[p] = image_aspect
            
        with open(f"{args.output_dir}/resnet152_roi_label.json", "w",encoding='utf-8') as f:
            json.dump(image_label_dict, f,indent = 2,ensure_ascii=False)

if __name__ == "__main__":
   main()