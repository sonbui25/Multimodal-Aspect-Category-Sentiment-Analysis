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
    Thực hiện Beam Search tối ưu cho FCMFSeq2Seq.
    """
    model.eval()
    
    # 1. Chuẩn bị Batch Dimension (nếu input là 1 sample)
    if enc_ids.dim() == 1:
        enc_ids = enc_ids.unsqueeze(0)
        enc_mask = enc_mask.unsqueeze(0)
        enc_type = enc_type.unsqueeze(0)
        add_mask = add_mask.unsqueeze(0)
        vis_embeds = vis_embeds.unsqueeze(0)
        roi_embeds = roi_embeds.unsqueeze(0)
        roi_coors = roi_coors.unsqueeze(0)

    # 2. CHẠY ENCODER MỘT LẦN DUY NHẤT (Encoder Caching)
    # Thay vì gọi model() lặp lại, ta chỉ encoding 1 lần.
    with torch.no_grad():
        # Unpack tuple chỉ lấy hidden states (bỏ qua attentions ở vị trí thứ 2)
        encoder_results = model.encoder(
            enc_ids, 
            vis_embeds, 
            roi_embeds, 
            roi_coors, 
            enc_type, 
            enc_mask, 
            add_mask
        )
        
        # Kiểm tra nếu kết quả là tuple (do code mới trả về thêm attention)
        if isinstance(encoder_results, tuple):
            enc_outputs = encoder_results[0]
        else:
            enc_outputs = encoder_results
    
    # 3. Khởi tạo Beam
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id 

    # Tạo decoder input ban đầu
    decoder_input = torch.tensor([[start_token_id]], device=device, dtype=torch.long)
    
    # Khởi tạo state cho decoder (KV cache)
    # Lưu ý: Hàm init_state cần enc_outputs và valid_lens (ở đây valid_lens=None khi eval)
    state = model.decoder.init_state(enc_outputs, None)

    # Beam chứa: (score, sequence, state)
    # Score là log-likelihood (bắt đầu bằng 0)
    beams = [(0.0, decoder_input, state)]
    
    final_candidates = []

    for step in range(max_len):
        candidates = []
        
        # Duyệt qua các beam hiện tại
        for score, seq, current_state in beams:
            # Kiểm tra nếu đã kết thúc (gặp SEP)
            if seq[0, -1].item() == tokenizer.sep_token_id:
                final_candidates.append((score, seq))
                continue
            
            # CHỈ ĐƯA VÀO TOKEN CUỐI CÙNG để tận dụng KV Cache (incremental decoding)
            # Nếu model.decoder hỗ trợ state update đúng cách.
            # Tuy nhiên, code IAOGDecoder của  ghép chuỗi trong state: 
            # key_values = torch.cat((state[2][self.i], X), dim=1)
            # Nên ta chỉ cần đưa token mới nhất vào (dec_input_step).
            
            dec_input_step = seq[:, -1:] # Lấy token cuối cùng [1, 1]

            with torch.no_grad():
                # Gọi trực tiếp Decoder
                # Lưu ý: Cần deepcopy state nếu state là list mutable để tránh ảnh hưởng các beam khác
                # Nhưng PyTorch tensor trong list thì cần cẩn thận.
                # Để đơn giản và an toàn bộ nhớ, ta clone state cho mỗi candidate.
                
                # Copy state cho nhánh beam này (vì mỗi beam có lịch sử KV cache riêng)
                # State structure: [enc_outputs, enc_valid_lens, [layer_kv_cache...]]
                # Layer cache là List các Tensor.
                state_copy = [
                    current_state[0], # enc_outputs (share chung, ko cần copy deep)
                    current_state[1], # valid_lens
                    [layer_cache.clone() if layer_cache is not None else None for layer_cache in current_state[2]]
                ]
                
                # Forward Decoder
                # dec_input_step là token vừa sinh ra
                logits = model.decoder(dec_input_step, state_copy, is_train=False)
                # Logits shape: [Batch, 1, Vocab] -> lấy token cuối
                next_token_logits = logits[0, -1, :]
                
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Lấy Top-K
                top_k_scores, top_k_ids = torch.topk(log_probs, beam_size)
                
                for k in range(beam_size):
                    new_score = score + top_k_scores[k].item()
                    new_token = top_k_ids[k].view(1, 1)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    
                    # Lưu candidate gồm: score, chuỗi mới, và state mới (đã update sau forward)
                    candidates.append((new_score, new_seq, state_copy))
        
        # Pruning: Chọn top beam_size ứng viên tốt nhất
        if not candidates:
            break
            
        # Sắp xếp giảm dần theo score
        ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
        
        # Vì state_copy trong loop trên có thể bị reference chéo nếu implement không kỹ
        # ở đây mỗi candidate đã giữ state_copy riêng sau khi forward, nên an toàn.
        beams = ordered[:beam_size]
        
        # Early stopping nếu tất cả beams đều đã xong
        if all(seq[0, -1].item() == tokenizer.sep_token_id for _, seq, _ in beams):
            final_candidates.extend([(s, seq) for s, seq, _ in beams])
            break
    
    # Nếu chưa có candidate nào hoàn thành (vẫn đang chạy đến max_len)
    if len(final_candidates) == 0:
         final_candidates = [(s, seq) for s, seq, _ in beams]

    # Chọn câu tốt nhất
    best_score, best_seq = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0]
    
    pred_text = tokenizer.decode(best_seq[0], skip_special_tokens=True)
    
    # Clean text (nếu có lỗi tokenizer sinh ra khoảng trắng thừa)
    pred_text = pred_text.strip()
    
    return [pred_text]
'''
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
        sequence_output, pooled_output, enc_attentions = self.bert(input_ids, token_type_ids, attention_mask)
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
        print("Fusion Size:", fusion.size())
        # Create mask for fusion layer (1 + Num_Img + Num_Img)
        comb_attention_mask = added_attention_mask[:, :1 + self.num_imgs * 2]
        extended_attention_mask = comb_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Final Multimodal Encoding
        final_multimodal_encoder = self.mm_attention(fusion, extended_attention_mask)
        final_multimodal_encoder = final_multimodal_encoder[-1] 
        print("Final Multimodal Encoder Size:", final_multimodal_encoder.size())

        return final_multimodal_encoder
'''
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
        sequence_output, pooled_output, enc_attentions = self.bert(input_ids, token_type_ids, attention_mask)
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

        # sequence_output: [Batch, seq_len, Hidden] (seq_len linh hoạt)
        # all_h_i_features: [Batch, num_imgs, Hidden] (Ví dụ 7)
        # all_r_i_features: [Batch, num_imgs, Hidden] (Ví dụ 7)

        # 1. FUSION: Nối chuỗi động
        # Kết quả: [Batch, seq_len + 14, Hidden]
        fusion = torch.cat((sequence_output, all_h_i_features, all_r_i_features), dim=1)
        # 2. TẠO MASK ĐỘNG (Dynamic Masking)
        # Lấy độ dài thực tế của text trong batch hiện tại
        current_seq_len = sequence_output.size(1)
        
        # Cắt mask text tương ứng với độ dài thực
        # added_attention_mask gốc thường rất dài (max_len), ta cắt lấy phần text
        text_mask = added_attention_mask[:, :current_seq_len]
        
        # Tạo mask cho phần Visual (Visual luôn valid nên là 1)
        # Số lượng token visual = số lượng ảnh + số lượng ROI group
        num_visual_tokens = all_h_i_features.size(1) + all_r_i_features.size(1)
        
        visual_mask = torch.ones((text_mask.size(0), num_visual_tokens), 
                                 device=text_mask.device, dtype=text_mask.dtype)

        # Nối mask lại: [Batch, seq_len + 14]
        comb_attention_mask = torch.cat((text_mask, visual_mask), dim=1)

        # Convert sang format cho BERT Attention (Batch, 1, 1, Total_Len)
        extended_attention_mask = comb_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Encoding cuối cùng
        final_multimodal_encoder = self.mm_attention(fusion, extended_attention_mask)
        
        # Trả về chuỗi token đầy đủ (không lấy [-1] nữa nếu muốn Decoder nhìn thấy hết)
        # Nếu self.mm_attention là BertEncoder (nhiều lớp), nó trả về list các layer.
        # Ta lấy layer cuối cùng:
        # in ra kich thuoc cua final_multimodal_encoder
        return final_multimodal_encoder[-1], enc_attentions

class FCMFSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len_decoder, pretrained_hf_path, num_imgs, num_roi, alpha):
        super(FCMFSeq2Seq, self).__init__()
        self.encoder = FCMFEncoder(pretrained_hf_path, num_imgs=num_imgs, num_roi=num_roi, alpha=alpha)
        self.decoder = IAOGDecoder(vocab_size=vocab_size, max_len_decoder=max_len_decoder)
        self.num_imgs = num_imgs
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
        enc_output, enc_attentions = self.encoder(
            enc_X, 
            visual_embeds_att, 
            roi_embeds_att, 
            roi_coors, 
            token_type_ids, 
            attention_mask, 
            added_attention_mask
        )
        
        # 2. Tái tạo Mask Động để truyền cho Decoder
        # Vì Decoder cần biết trong enc_output chỗ nào là Text Padding
        
        # Lấy lại độ dài text thực tế từ output của encoder
        # Tổng length - 14 (Visual) = Text Length
        num_visual_tokens = self.num_imgs * 2 # Hoặc lấy động từ encoder nếu cấu trúc đổi
        current_text_len = enc_output.size(1) - num_visual_tokens
        
        # Lấy mask gốc từ input
        text_mask = attention_mask[:, :current_text_len] # [Batch, seq_len]
        
        # Tạo mask visual
        vis_mask = torch.ones((text_mask.size(0), num_visual_tokens), 
                              device=text_mask.device, dtype=text_mask.dtype)
        
        # Combined Mask: [Batch, seq_len + 14]
        combined_mask = torch.cat((text_mask, vis_mask), dim=1)

        # [FIX HERE] Tạo state đủ 3 thành phần: [Encoder_Out, Mask, Cache_List]
        # Cache_List cần thiết cho TransformerDecoderBlock (state[2])
        dec_state = [enc_output, combined_mask, [None] * self.decoder.num_blks]

        # 3. Gọi Decoder với state đã sửa
        logits = self.decoder(dec_X, 
                              dec_state, 
                              is_train=is_train)
        if not is_train:
            return logits, enc_attentions
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