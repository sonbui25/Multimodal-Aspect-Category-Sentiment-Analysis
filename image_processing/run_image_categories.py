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
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from torchvision.models import resnet152,ResNet152_Weights,resnet50, ResNet50_Weights
import random
import logging
import sys
from collections import defaultdict

class ImageDataset(Dataset):
    def __init__(self, data,root_dir):
        self.image_label = data
        self.root_dir = root_dir
    def __len__(self):
        return self.image_label.shape[0]
    def __getitem__(self,index):
        image_name = self.image_label.loc[index, "file_name"]
        
        image = torchvision.io.read_image(os.path.join(self.root_dir,image_name),mode = torchvision.io.ImageReadMode.RGB)

        label = torch.from_numpy(self.image_label.iloc[index,2:].values.astype(int)) 

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
            "label": label,
            "filename": image_name 
        }
    
class MyImgModel(torch.nn.Module):
  def __init__(self,num_classes):
    super(MyImgModel, self).__init__()
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
      = precision_recall_fscore_support(y_true, y_pred, average='macro',zero_division=0.0,labels = [0,1])
    return p_macro, r_macro, f_macro

def convert_img_to_tensor(root_dir, img_path):
    image = torchvision.io.read_image(os.path.join(root_dir,img_path),mode = torchvision.io.ImageReadMode.RGB)
    transforms = v2.Compose([
                        v2.Resize((224,224),antialias=True),
                        v2.ConvertImageDtype(torch.float32),
                        v2.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                                    ])
    image = transforms(image)
    return image

def predict_wrapper(model, list_aspect, root_dir, img_path,device):
    img = convert_img_to_tensor(root_dir, img_path)
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred).squeeze(0)
    pred = pred > 0.45 
    pred = pred.cpu().numpy().astype(int)
    pred = np.where(pred==1)[0]
    return list(list_aspect[pred])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='../image', type=str, required=True)
    parser.add_argument("--image_label_path", default=None, type=str)
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
    print("===================== RUN IMAGE CATEGORIES =====================")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(
        format=log_format,
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{args.output_dir}/image_categories.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    if args.no_cuda: device = 'cpu'
    else: device = 'cuda'

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if not args.do_train and not args.get_cate:
        raise ValueError("At least one of `do_train` or `get_cate` must be True.")
    
    ASPECT = ['Food', 'Room', 'Facilities', 'Service', 'Public_area'] 

    if args.do_train:
        if args.image_label_path == None: raise ValueError("Please provide annotated image file.")

        image_label = pd.read_excel(args.image_label_path)
        image_label = image_label.fillna(0)
        image_label = image_label.loc[~(image_label.iloc[:,1:]==0).all(axis=1)]
        image_label = image_label.reset_index().drop("index",axis=1)

        train_data, dev_test_data = train_test_split(image_label,test_size=0.3,random_state=18)
        dev_data, test_data = train_test_split(dev_test_data,test_size=0.5,random_state=18)

        train_data = train_data.reset_index().drop('index',axis=1)
        dev_data = dev_data.reset_index().drop('index',axis=1)
        test_data = test_data.reset_index().drop('index',axis=1)

        train_set = ImageDataset(train_data,args.image_dir)
        dev_set = ImageDataset(dev_data,args.image_dir)
        test_set = ImageDataset(test_data,args.image_dir)

        train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle = True)
        dev_loader = DataLoader(dev_set,batch_size=args.eval_batch_size,shuffle = False)
        test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle = False)

        model = MyImgModel(len(ASPECT)) 
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
        max_accracy = 0.0

        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_loader, position=0, leave=False, desc="Train")):
                input = batch['image'].to(device)
                label = batch['label'].to(device)
                logits = model(input)
                loss = criterion(logits,label.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logger.info("***** Running evaluation on Dev Set*****")
            model.eval()
            idx2asp = {i:v for i,v in enumerate(ASPECT)}
            true_label_list = {asp:[] for asp in ASPECT}
            pred_label_list = {asp:[] for asp in ASPECT}

            for step, batch in enumerate(tqdm(dev_loader, position=0, leave=True, desc="Dev")):
                input = batch['image'].to(device)
                label = batch['label'].to(device)
                with torch.no_grad():
                    logits = model(input)
                    logits = torch.sigmoid(logits).cpu().numpy()
                    
                    for id_asp in range(len(ASPECT)):
                        asp_label = label[:,id_asp].cpu().numpy()
                        pred = np.asarray(logits[:,id_asp] > 0.7).astype(int)
                        true_label_list[idx2asp[id_asp]].append(asp_label)
                        pred_label_list[idx2asp[id_asp]].append(pred)

            all_precision, all_recall, all_f1 = 0, 0, 0
            all_accuracy = 0
            for id_asp in range(len(ASPECT)):
                tr = np.concatenate(true_label_list[idx2asp[id_asp]])
                pr = np.concatenate(pred_label_list[idx2asp[id_asp]])
                precision, recall, f1_score = macro_f1(tr,pr)
                accuracy = accuracy_score(tr,pr)
                all_precision += precision; all_recall += recall; all_f1 += f1_score; all_accuracy += accuracy

            all_accuracy /= len(ASPECT)
            
            if all_accuracy >= max_accracy:  
                save_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth',model,train_idx)
                max_accracy = all_accuracy
                logger.info(f"New Best Accuracy: {max_accracy:.4f}")

        # --- TEST SECTION (FIXED ORDER & FORMAT) ---
        output_test_file = os.path.join(args.output_dir, "test_image_results.txt")
        output_detail_file = os.path.join(args.output_dir, "test_image_predictions_detail.txt")
        
        logger.info("***** Running evaluation on Test Set *****")
        checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        true_label_list = {asp:[] for asp in ASPECT}
        pred_label_list = {asp:[] for asp in ASPECT}
        
        # Dùng set để gom nhóm aspect unique (phòng trường hợp 1 ảnh bị lặp dòng)
        results_map = defaultdict(lambda: {"gold": set(), "pred": set()})

        for step, batch in enumerate(tqdm(test_loader, position=0, leave=True, desc="Test")):
            input = batch['image'].to(device)
            label = batch['label'].to(device)
            filenames = batch['filename']
            
            with torch.no_grad():
                logits = model(input)
                logits = torch.sigmoid(logits).cpu().numpy() 
                label_np = label.cpu().numpy()               

                # Metric calculation standard logic
                for id_asp in range(len(ASPECT)):
                    asp_label = label[:,id_asp].cpu().numpy()
                    pred = np.asarray(logits[:,id_asp] > 0.7).astype(int) 
                    true_label_list[idx2asp[id_asp]].append(asp_label)
                    pred_label_list[idx2asp[id_asp]].append(pred)

                # --- Aggregation Logic ---
                batch_size = input.shape[0]
                threshold = 0.7
                
                for i in range(batch_size):
                    fname = filenames[i]
                    pred_indices = np.where(logits[i] > threshold)[0]
                    gold_indices = np.where(label_np[i] == 1)[0]
                    
                    pred_aspects = [ASPECT[idx] for idx in pred_indices]
                    gold_aspects = [ASPECT[idx] for idx in gold_indices]
                    
                    results_map[fname]["gold"].update(gold_aspects)
                    results_map[fname]["pred"].update(pred_aspects)

        # Ghi Metrics chung
        with open(output_test_file, "w", encoding='utf-8') as writer:
            writer.write("***** TEST RESULTS (Image Categories) *****\n")
            test_all_f1, test_all_accuracy = 0, 0
            for id_asp in range(len(ASPECT)):
                tr = np.concatenate(true_label_list[idx2asp[id_asp]])
                pr = np.concatenate(pred_label_list[idx2asp[id_asp]])
                precision, recall, f1_score = macro_f1(tr,pr)
                accuracy = accuracy_score(tr,pr)
                writer.write(f"{idx2asp[id_asp]:<20} | F1: {f1_score:.4f} | Acc: {accuracy:.4f}\n")
                test_all_f1 += f1_score; test_all_accuracy += accuracy

            test_all_f1 /= len(ASPECT); test_all_accuracy /= len(ASPECT)
            writer.write(f"MACRO AVG | F1: {test_all_f1:.4f} | Acc: {test_all_accuracy:.4f}\n")

        # Ghi Detailed Log (Duyệt theo thứ tự UNIQUE trong dataframe gốc)
        test_ordered_files = test_data['file_name'].unique()
        
        with open(output_detail_file, "w", encoding='utf-8') as f:
            for fname in test_ordered_files:
                if fname in results_map:
                    content = results_map[fname]
                    # Sort để aspect 1 luôn đứng trước aspect 2
                    sorted_gold = sorted(list(content["gold"]))
                    sorted_pred = sorted(list(content["pred"]))
                    
                    # Format chính xác như yêu cầu
                    f.write(f'"{fname}": [\n')
                    f.write(f'    Gold_Label: {json.dumps(sorted_gold, ensure_ascii=False)},\n')
                    f.write(f'    Prediction: {json.dumps(sorted_pred, ensure_ascii=False)},\n')
                    f.write('  ],\n')
        
        logger.info(f"Saved detailed predictions (Quantity: {len(test_ordered_files)}) to {output_detail_file}")

    if args.get_cate:
        print("===================== GET IMAGE CATEGORIES =====================")
        model = MyImgModel(len(ASPECT)) 
        if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
        model = model.to(device)

        if args.do_train: checkpoint = load_model(f'{args.output_dir}/seed_{args.seed}_image_model.pth')
        else: checkpoint = load_model(args.weight_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        ASPECT = np.asarray(ASPECT)
        image_path_list = os.listdir(args.image_dir)
        img_label_dict = {}
        
        for img_path in tqdm(image_path_list, desc="Inferencing"):
            lb = predict_wrapper(model, ASPECT,args.image_dir, img_path,device)
            img_label_dict[img_path] = lb

        with open(f"{args.output_dir}/resnet152_image_label.json", "w",encoding='utf-8') as f:
            json.dump(img_label_dict, f,indent = 2,ensure_ascii=False)

if __name__ == "__main__":
   main()