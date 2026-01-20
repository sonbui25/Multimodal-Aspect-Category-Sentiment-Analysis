import torch
import argparse
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from matplotlib import font_manager
from transformers import AutoTokenizer
from torchvision.models import resnet152, ResNet152_Weights

# Suppress font glyph warning
warnings.filterwarnings('ignore', message='.*Glyph.*missing from current font.*')

# Import modules từ framework
from fcmf_framework.fcmf_pretraining import FCMFSeq2Seq, beam_search
from fcmf_framework.resnet_utils import myResNetImg, myResNetRoI
from text_preprocess import TextNormalize, convert_unicode
from underthesea import text_normalize
from iaog_dataset import IAOGDataset

# Cấu hình logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- BIẾN TOÀN CỤC ---
captured_self_attn = []

def self_attn_hook(module, input, output):
    """Bắt lấy attention probs"""
    if isinstance(output, tuple) and len(output) > 1:
        captured_self_attn.append(output[1].detach().cpu())

# --- 1. HELPERS ---
def setup_vietnamese_font():
    preferred_fonts = ['Arial']
    system_fonts = [f.name for f in font_manager.fontManager.ttflist]
    selected_font = next((f for f in preferred_fonts if f in system_fonts), 'DejaVu Sans')
    plt.rcParams['font.family'] = selected_font
    return selected_font

def clean_word_display(word):
    """Làm sạch và hiển thị từ, giữ lại token đặc biệt"""
    # Nếu là token đặc biệt, giữ lại
    special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '[CLS]', '[SEP]', '[START]', '[END]']
    if word in special_tokens:
        return word
    
    # Xóa marker sub-word và space
    cleaned = word.replace('\u2581', '').replace(' ', '').strip()
    
    # Nếu sau khi xóa mà rỗng, trả về token gốc
    if not cleaned:
        return word if word else '?'
    
    return cleaned

def aggregate_attention(attention_map, x_raw_tokens, y_raw_tokens):
    """Gộp token sub-word thành word"""
    SP_MARKER = "\u2581"
    def group_subwords_smart(tokens):
        groups, labels, curr_idx, curr_word = [], [], [], ""
        for i, token in enumerate(tokens):
            if token in ['<s>', '</s>', '<pad>', '[START]', '[END]', '[CLS]', '[SEP]']:
                if curr_idx: groups.append(curr_idx); labels.append(clean_word_display(curr_word))
                groups.append([i]); labels.append(token); curr_idx, curr_word = [], ""
                continue
            has_marker = token.startswith(SP_MARKER) or token.startswith('_')
            is_new_word = has_marker or (not curr_idx and not token.startswith('<'))
            if is_new_word:
                if curr_idx: groups.append(curr_idx); labels.append(clean_word_display(curr_word))
                curr_idx, curr_word = [i], token
            else:
                curr_idx.append(i); curr_word += token
        if curr_idx: groups.append(curr_idx); labels.append(clean_word_display(curr_word))
        return groups, labels

    y_groups, y_labels = group_subwords_smart(y_raw_tokens)
    x_groups, x_labels = group_subwords_smart(x_raw_tokens)
    new_attn = np.zeros((len(y_groups), len(x_groups)))
    for i, y_grp in enumerate(y_groups):
        for j, x_grp in enumerate(x_groups):
            sub_matrix = attention_map[np.ix_(y_grp, x_grp)]
            new_attn[i, j] = np.sum(sub_matrix) / max(len(y_grp), 1)
    return new_attn, x_labels, y_labels

def extract_gold_labels_for_aspect(raw_labels, aspect):
    """Trích xuất opinion words cho aspect"""
    gold_words = []
    aspect_norm = aspect.replace(" ", "_").lower()
    for lbl in raw_labels:
        if '#' in lbl:
            word, lbl_aspect = lbl.split('#')
            lbl_aspect_norm = lbl_aspect.strip().replace("Public area", "Public_area").lower()
            if lbl_aspect_norm == aspect_norm:
                gold_words.append(word.strip())
    return gold_words

# --- 2. HÀM VẼ FIRST HEAD CỦA 12 LAYERS TRÊN 1 FIGURE ---
def plot_all_heads_of_layers(attention_list, x_tokens, y_tokens, title, save_path, cmap='viridis'):
    """
    Vẽ heatmap của head đầu tiên của tất cả 12 layers trên 1 figure (3x4 grid).
    attention_list: List of layer attention tensors, mỗi cái shape [Batch, 12, Seq, Seq]
    """
    setup_vietnamese_font()
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    for layer_idx, attention_tensor in enumerate(attention_list):
        if layer_idx >= 12:
            break
        
        # attention_tensor shape: [Batch, 12, Seq, Seq]
        # Move to CPU if on GPU
        if hasattr(attention_tensor, 'cpu'):
            attention_tensor = attention_tensor.cpu()
        
        layer_tensor = attention_tensor.squeeze(0) # [12, Seq, Seq]
        
        # Lấy head 0
        if x_tokens is y_tokens:  # Self-attention
            first_head = layer_tensor[0, :len(x_tokens), :len(x_tokens)].numpy()
        else:  # Cross-attention
            first_head = layer_tensor[0, :len(y_tokens), :len(x_tokens)].numpy()
        
        # Aggregate
        agg_w, x_lbl, y_lbl = aggregate_attention(first_head, x_tokens, y_tokens)
        
        ax = axes[layer_idx]
        sns.heatmap(
            agg_w, xticklabels=x_lbl, yticklabels=y_lbl,
            cmap=cmap, annot=True, fmt='.2f', annot_kws={'size': 5, 'weight': 'bold'}, 
            cbar=True, ax=ax, square=False, cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f"Layer {layer_idx + 1} - Head 0", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    
    # Hide unused subplots
    for idx in range(len(attention_list), 12):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=22, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved all 12 layers visualization to: {save_path}")


# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--pretrained_hf_model", default="uitnlp/visobert", type=str)
    parser.add_argument("--test_file", required=True, type=str)
    parser.add_argument("--sample_index", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_len_decoder", type=int, default=10)
    parser.add_argument("--max_seq_length", default=170, type=int)
    parser.add_argument("--num_imgs", default=7, type=int)
    parser.add_argument("--num_rois", default=7, type=int)
    parser.add_argument("--fine_tune_cnn", action='store_true')
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--IAOG_visualize_dir", default="./IAOG_Heads_Vis", type=str)
    # [NEW] Tham số chọn Layer muốn soi
    parser.add_argument("--target_layer", type=int, default=0, help="Layer index to visualize (0-11)")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.IAOG_visualize_dir, exist_ok=True)

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_hf_model)
    model = FCMFSeq2Seq(len(tokenizer), args.max_len_decoder, args.pretrained_hf_model, args.num_imgs, args.num_rois, args.alpha)
    model.encoder.bert.cell.resize_token_embeddings(len(tokenizer))
    model.encoder.bert.cell.config.output_attentions = True 
    model.decoder.embedding = torch.nn.Embedding(len(tokenizer), model.decoder.num_hiddens)
    model.to(device)

    for layer in model.encoder.bert.cell.encoder.layer:
        layer.attention.self.register_forward_hook(self_attn_hook)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    if missing_keys:
        print(f"[WARNING] Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    model.eval()

    # ResNets & Dataset
    img_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    roi_res = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(device)
    resnet_img = myResNetImg(img_res, args.fine_tune_cnn, device).to(device)
    resnet_roi = myResNetRoI(roi_res, args.fine_tune_cnn, device).to(device)
    
    roi_df = pd.read_csv(os.path.join(args.data_dir, "roi_data.csv"))
    roi_df['file_name'] = roi_df['file_name'] + '.png'
    with open(os.path.join(args.data_dir, 'resnet152_image_label.json')) as f: dict_image_aspect = json.load(f)
    with open(os.path.join(args.data_dir, 'resnet152_roi_label.json')) as f: dict_roi_aspect = json.load(f)
    
    with open(args.test_file, 'r', encoding='utf-8') as f: data_json = json.load(f)
    sample_raw = data_json[args.sample_index]
    dummy_df = pd.DataFrame([sample_raw])
    iaog_ds_helper = IAOGDataset(dummy_df, tokenizer, args.image_dir, roi_df, dict_image_aspect, dict_roi_aspect, args.num_imgs, args.num_rois)
    
    logger.info(f"Processing Sample {args.sample_index}. Visualizing HEADS of Layer {args.target_layer}")

    list_img_path = sample_raw.get('list_img', [])
    t_img, roi_img, roi_coors = iaog_ds_helper._process_images(list_img_path)
    with torch.no_grad():
        vis_embeds = torch.stack([resnet_img(t_img.to(device).unsqueeze(0)[:,i]).view(-1,2048,49).permute(0,2,1) for i in range(args.num_imgs)], dim=1)
        roi_embeds = torch.stack([torch.stack([resnet_roi(roi_img.to(device).unsqueeze(0)[:,i,r].float()).squeeze(1) for r in range(args.num_rois)], dim=1) for i in range(args.num_imgs)], dim=1)
    roi_coors = roi_coors.to(device).unsqueeze(0)

    # Aspects Loop
    raw_labels = sample_raw.get('iaog_labels', sample_raw.get('labels', []))
    active_aspects = set()
    for lbl in raw_labels:
        if '#' in lbl: active_aspects.add(lbl.split('#')[1].strip().replace("Public area", "Public_area"))
    if not active_aspects: active_aspects.add("Room")

    for aspect in sorted(list(active_aspects)):
        logger.info(f"--- Processing Aspect: {aspect} ---")
        captured_self_attn.clear()
        
        text_content = TextNormalize().normalize(text_normalize(convert_unicode(sample_raw['comment'])))
        list_image_aspect, list_roi_aspect = iaog_ds_helper._get_visual_tags(list_img_path)
        joined_tags = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}".lower().replace('_', ' ')
        asp_text = aspect.replace("_", " ")
        combine_text = f"{asp_text} </s></s> {text_content}".lower()
        
        enc = tokenizer(combine_text, joined_tags, max_length=args.max_seq_length, truncation='only_first', padding='max_length', return_token_type_ids=True, return_tensors='pt')
        enc_ids = enc['input_ids'].to(device).squeeze(0)
        enc_mask = enc['attention_mask'].to(device).squeeze(0)
        enc_type = enc['token_type_ids'].to(device).squeeze(0)
        add_mask = torch.tensor([1] * (args.max_seq_length + 49)).to(device).squeeze(0)

        # Predict
        pred_text = beam_search(model, tokenizer, enc_ids, enc_mask, enc_type, add_mask, vis_embeds.squeeze(0), roi_embeds.squeeze(0), roi_coors.squeeze(0), beam_size=args.beam_size, max_len=args.max_len_decoder, device=device)[0]
        if not pred_text or pred_text == "none": continue
        
        captured_self_attn.clear()
        
        # Forward Pass
        bos_token = tokenizer.cls_token 
        dec_input = tokenizer(f"{bos_token} {pred_text}", return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        
        with torch.no_grad():
            _ = model(enc_X=enc_ids.unsqueeze(0), dec_X=dec_input, visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors, token_type_ids=enc_type.unsqueeze(0), attention_mask=enc_mask.unsqueeze(0), added_attention_mask=add_mask.unsqueeze(0), is_train=False)

        real_enc_len = torch.sum(enc_mask).item()
        x_tokens = tokenizer.convert_ids_to_tokens(enc_ids)[:real_enc_len]
        y_tokens = tokenizer.convert_ids_to_tokens(dec_input.squeeze(0))[1:] 

        # --- VISUALIZE ALL 12 LAYERS HEAD 0 ON 1 FIGURE ---
        
        # G1. Encoder Self-Attention (12 layers trên 1 figure)
        if len(captured_self_attn) > 0:
            path_self = os.path.join(args.IAOG_visualize_dir, f"sample{args.sample_index}_{aspect}_Encoder_All12Layers_Head0.png")
            plot_all_heads_of_layers(captured_self_attn, x_tokens, x_tokens, 
                                     f"Encoder Self-Attention: All 12 Layers - Head 0", path_self)

        # G2. Decoder Cross-Attention (12 layers trên 1 figure)
        try:
            if hasattr(model.decoder, 'attention_weights') and len(model.decoder.attention_weights) > 1:
                cross_attn = model.decoder.attention_weights[1]
                if cross_attn is not None and len(cross_attn) > 0:
                    gold_words = extract_gold_labels_for_aspect(raw_labels, aspect)
                    gold_str = ", ".join(gold_words) if gold_words else "N/A"
                    
                    path_cross = os.path.join(args.IAOG_visualize_dir, f"sample{args.sample_index}_{aspect}_Decoder_All12Layers_Head0.png")
                    title_with_gold = f"Decoder Cross-Attention: All 12 Layers - Head 0\n[GOLD: {gold_str}] [PRED: {pred_text}]"
                    plot_all_heads_of_layers(cross_attn, x_tokens, y_tokens, title_with_gold, path_cross)
                else:
                    logger.warning(f"Cross-attention weights not found for aspect {aspect}")
            else:
                logger.warning(f"Decoder attention_weights not available for aspect {aspect}")
        except Exception as e:
            logger.warning(f"Error visualizing cross-attention for aspect {aspect}: {str(e)}")

    logger.info("Visualization complete.")

if __name__ == "__main__":
    main()