import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from .mm_modeling import *
from .roi_modeling import *
import torch
import torch.nn.functional as F
import math

# ==============================================================================
# HÀM BEAM SEARCH CHUẨN (Dựa trên Dive into Deep Learning 10.8)
# ==============================================================================
def beam_search(model, tokenizer, enc_ids, enc_mask, enc_type, add_mask, 
                vis_embeds, roi_embeds, roi_coors, 
                beam_size=3, num_preds=1, max_len=20, device='cuda'):
    """
    Thực hiện Beam Search cho mô hình FCMFSeq2Seq.
    
    Args:
        model: Mô hình FCMFSeq2Seq đã train.
        tokenizer: Tokenizer để lấy ID đặc biệt.
        enc_ids, ...: Các input cho Encoder (đã batch hóa hoặc 1 sample).
        beam_size (int): Kích thước chùm (k).
        num_preds (int): Số lượng câu dự đoán trả về (thường là 1 câu tốt nhất).
        max_len (int): Độ dài tối đa của câu sinh ra.
    
    Returns:
        decoded_preds (list): List các chuỗi văn bản đã decode.
    """
    model.eval()
    
    # 1. Chuẩn bị dữ liệu (Giả định đầu vào là Batch=1 để dễ hiểu và đúng chuẩn d2l)
    # Nếu đầu vào là batch lớn, vòng lặp eval bên ngoài sẽ gọi hàm này từng cái một.
    if enc_ids.dim() == 1:
        enc_ids = enc_ids.unsqueeze(0)
        enc_mask = enc_mask.unsqueeze(0)
        enc_type = enc_type.unsqueeze(0)
        add_mask = add_mask.unsqueeze(0)
        vis_embeds = vis_embeds.unsqueeze(0)
        roi_embeds = roi_embeds.unsqueeze(0)
        roi_coors = roi_coors.unsqueeze(0)

    # 2. Chạy Encoder một lần duy nhất (Encoder output được dùng lại)
    # Lưu ý: Model FCMF có thể forward cả encoder lẫn decoder cùng lúc.
    # Nhưng để tối ưu, ta nên tách ra nếu model hỗ trợ. 
    # Ở đây ta giả lập việc gọi model.encoder (nếu có) hoặc chạy forward đầy đủ.
    # Để an toàn với code hiện tại (chạy full forward), ta sẽ truyền input lặp lại.

    # 3. Khởi tạo Beam
    # Beam chứa: (score, sequence_tensor)
    # Score ban đầu là 0.0, sequence là [CLS]
    start_token = tokenizer.cls_token_id
    initial_seq = torch.tensor([[start_token]], device=device, dtype=torch.long)
    
    # Danh sách k ứng viên hiện tại (Score, Sequence)
    # Lưu ý: Score ở đây là Log-Likelihood
    beams = [(0.0, initial_seq)]
    
    # Danh sách các câu đã hoàn thành (gặp token [SEP])
    final_candidates = []

    for step in range(max_len):
        candidates = []
        
        # Duyệt qua từng ứng viên trong beam hiện tại
        for score, seq in beams:
            # Nếu câu đã kết thúc ở bước trước (nhưng chưa bị loại), ta add vào final luôn
            if seq[0, -1].item() == tokenizer.sep_token_id:
                final_candidates.append((score, seq))
                continue
                
            # Chuẩn bị input mở rộng (Expand inputs to match beam items if needed)
            # Ở đây ta chạy model forward cho 1 sequence này
            with torch.no_grad():
                # Forward Model
                # Lưu ý: Model trả về Logits [Batch, Seq_Len, Vocab]
                logits = model(enc_X=enc_ids, dec_X=seq,
                               visual_embeds_att=vis_embeds, roi_embeds_att=roi_embeds, roi_coors=roi_coors,
                               token_type_ids=enc_type, attention_mask=enc_mask, 
                               added_attention_mask=add_mask, is_train=False)
                
                # Lấy logit của từ cuối cùng
                next_token_logits = logits[0, -1, :]
                
                # Tính Log-Softmax để có Log-Probabilities chuẩn
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Lấy Top-K từ vựng có xác suất cao nhất cho nhánh này
                # (Lấy beam_size ứng viên)
                top_k_scores, top_k_ids = torch.topk(log_probs, beam_size)
                
                for k in range(beam_size):
                    # Cộng dồn điểm số: Score mới = Score cũ + LogProb mới
                    new_score = score + top_k_scores[k].item()
                    
                    # Ghép từ mới vào chuỗi
                    new_token = top_k_ids[k].view(1, 1)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    
                    candidates.append((new_score, new_seq))
        
        # --- Pruning (Tỉa nhánh) ---
        # 1. Sắp xếp tất cả ứng viên theo điểm số giảm dần (tốt nhất lên đầu)
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
        
        # 2. Chọn ra k ứng viên tốt nhất để làm beam cho vòng sau
        beams = ordered[:beam_size]
        
        # Tối ưu: Nếu tất cả top-k beams đều đã kết thúc, dừng sớm
        if all(seq[0, -1].item() == tokenizer.sep_token_id for _, seq in beams):
            break
            
    # Thêm các beam còn lại vào danh sách ứng viên cuối cùng (nếu chưa kết thúc)
    final_candidates.extend(beams)
    
    # Sắp xếp lại lần cuối và chọn câu tốt nhất
    # (Có thể áp dụng length penalty ở đây: score / len(seq)**alpha)
    best_score, best_seq = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0]
    
    # Decode ra text
    pred_text = tokenizer.decode(best_seq[0], skip_special_tokens=True)
    
    return [pred_text]
class FCMFEncoder(nn.Module):
    def __init__(self, pretrained_hf_path, num_imgs=7, num_roi=4, alpha=0.7):
        super(FCMFEncoder, self).__init__()
        self.num_imgs = num_imgs
        self.num_roi = num_roi
        self.alpha = alpha
        
        # Backbone BERT
        self.bert = FeatureExtractor(pretrained_hf_path)
        
        # Projections
        self.vismap2text = nn.Linear(2048, HIDDEN_SIZE)
        self.roimap2text = nn.Linear(2048, HIDDEN_SIZE)
        
        # Heads & Attention Modules
        self.box_head = BoxMultiHeadedAttention(8, HIDDEN_SIZE)
        self.text2img_attention = BertCrossEncoder()
        self.text2img_pooler = BertPooler()
        self.text2roi_pooler = BertPooler()
        
        # [NEW] Multimodal Denoising Encoder (MDE)
        self.MultimodalDenoisingEncoder = MultimodalDenoisingEncoder(alpha=alpha)
        
        self.mm_attention = MultimodalEncoder()

    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors=None, token_type_ids=None, attention_mask=None, added_attention_mask=None):
        # 1. Text Encoding
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        seq_len = sequence_output.size()[1]

        list_h_i = []
        list_r_i = []
        
        for i in range(self.num_imgs):
            # --- A. IMAGE-GUIDED ATTENTION ---
            one_img_embeds = visual_embeds_att[:, i, :] # [Batch, 49, 2048]
            converted_img_embed_map = self.vismap2text(one_img_embeds) # [Batch, 49, Hidden]

            # Prepare Original Mask for 49 patches
            img_mask_orig = added_attention_mask[:, :49]
            extended_img_mask_orig = img_mask_orig.unsqueeze(1).unsqueeze(2)
            extended_img_mask_orig = extended_img_mask_orig.to(dtype=converted_img_embed_map.dtype)
            extended_img_mask_orig = (1.0 - extended_img_mask_orig) * -10000.0

            # --- B. MULTIMODAL DENOISING (MDE) ---
            if self.alpha < 1.0:
                # Apply MDE to filter weak patches based on text guidance
                image_features_denoised = self.MultimodalDenoisingEncoder(
                    sequence_output, 
                    converted_img_embed_map
                )
                
                # Create NEW Mask for the filtered patches (k patches)
                k_size = image_features_denoised.size(1) 
                batch_size = image_features_denoised.size(0)
                device = image_features_denoised.device
                
                # Assume all filtered patches are valid (1)
                new_img_mask = torch.ones((batch_size, k_size), device=device)
                
                # Extend mask for Attention (1.0 for valid, -10000.0 for masked)
                extended_new_mask = new_img_mask.unsqueeze(1).unsqueeze(2)
                extended_new_mask = extended_new_mask.to(dtype=image_features_denoised.dtype)
                extended_new_mask = (1.0 - extended_new_mask) * -10000.0
            else:
                # No denoising, use full features
                image_features_denoised = converted_img_embed_map
                extended_new_mask = extended_img_mask_orig

            # --- C. CROSS-MODAL ATTENTION ---
            text2img_cross_attention = self.text2img_attention(
                sequence_output, 
                image_features_denoised, 
                extended_new_mask
            )
            text2img_output_layer = text2img_cross_attention[-1]
            text2img_cross_output = self.text2img_pooler(text2img_output_layer) 
            transpose_text2img_embed = text2img_cross_output.unsqueeze(1) 

            list_h_i.append(transpose_text2img_embed) 

            # --- D. GEOMETRIC ROI-AWARE ATTENTION ---
            # ROI Mask processing
            text2roi_mask = added_attention_mask[:, :seq_len + self.num_roi]
            text2roi_mask = text2roi_mask.unsqueeze(1).unsqueeze(2)
            text2roi_mask = text2roi_mask.to(dtype=text2roi_mask.dtype)
            text2roi_mask = (1.0 - text2roi_mask) * -10000.0

            roi_at_i_img = roi_embeds_att[:, i, :]
            converted_roi_embed_map = self.roimap2text(roi_at_i_img)
            
            # Calculate geometric relations
            relative_roi = self.box_head(
                converted_roi_embed_map,
                converted_roi_embed_map,
                converted_roi_embed_map,
                roi_coors[:, i, :]
            )

            # Concatenate Text + ROI relations
            text_roi_output = torch.cat((sequence_output, relative_roi), dim=1) 

            # Multimodal Self-Attention
            roi_multimodal_encoder = self.mm_attention(text_roi_output, text2roi_mask)
            roi_att_text_output_layer = roi_multimodal_encoder[-1]
            
            # Pooling
            roi_pooling = self.text2roi_pooler(roi_att_text_output_layer)
            transpose_roi_embed = roi_pooling.unsqueeze(1) 

            list_r_i.append(transpose_roi_embed) 

        # Combine all features
        all_h_i_features = torch.cat(list_h_i, dim=1) # [Batch, Num_Img, Hidden]
        all_r_i_features = torch.cat(list_r_i, dim=1) # [Batch, Num_Img, Hidden]

        # Fusion: [CLS] + Image Features + ROI Features
        fusion = torch.cat((sequence_output[:, 0, :].unsqueeze(1), all_h_i_features, all_r_i_features), dim=1)
    
        # Create mask for fusion layer (1 + Num_Img + Num_Img)
        comb_attention_mask = added_attention_mask[:, :1 + self.num_imgs * 2]
        extended_attention_mask = comb_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Final Multimodal Encoding
        final_multimodal_encoder = self.mm_attention(fusion, extended_attention_mask)
        final_multimodal_encoder = final_multimodal_encoder[-1] 
            
        return final_multimodal_encoder

class FCMFSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len_decoder, pretrained_hf_path, num_imgs, num_roi, alpha):
        super(FCMFSeq2Seq, self).__init__()
        self.encoder = FCMFEncoder(pretrained_hf_path, num_imgs=num_imgs, num_roi=num_roi, alpha=alpha)
        self.decoder = IAOGDecoder(vocab_size=vocab_size, max_len_decoder=max_len_decoder)
        
        # --- Weight Initialization ---
        self.decoder.apply(self._init_weights)
        self.encoder.vismap2text.apply(self._init_weights)
        self.encoder.roimap2text.apply(self._init_weights)
        self.encoder.box_head.apply(self._init_weights)
        self.encoder.text2img_attention.apply(self._init_weights)
        self.encoder.mm_attention.apply(self._init_weights)
        self.encoder.MultimodalDenoisingEncoder.apply(self._init_weights)

        # --- Weight Tying (Embeddings) ---
        if hasattr(self.encoder.bert.cell, 'embeddings'):
             self.decoder.embedding.weight = self.encoder.bert.cell.embeddings.word_embeddings.weight
        
        self.decoder.dense.weight = self.decoder.embedding.weight

    def forward(self, enc_X, dec_X, visual_embeds_att, roi_embeds_att, roi_coors=None, token_type_ids=None, attention_mask=None, added_attention_mask=None, source_valid_len=None, is_train=True):
        enc_output_last_layer = self.encoder(
            enc_X, 
            visual_embeds_att, 
            roi_embeds_att, 
            roi_coors, 
            token_type_ids, 
            attention_mask, 
            added_attention_mask
        )
        dec_state = self.decoder.init_state(enc_output_last_layer, source_valid_len)
        logits = self.decoder(dec_X, dec_state, is_train=is_train)
        return logits

    def _init_weights(self, module):
        """Initialize weights with Normal distribution (std=0.02)"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()