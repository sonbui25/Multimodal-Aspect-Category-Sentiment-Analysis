import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from .mm_modeling import *
from .roi_modeling import *
from .fcmf_pretraining import FCMFEncoder
from .roi_modeling import *
from .fcmf_pretraining import FCMFEncoder
class FCMF(nn.Module):
    def __init__(self, pretrained_path, num_labels=4, num_imgs = 7, num_roi = 7, alpha=0.7):
        super(FCMF, self).__init__()
        self.encoder = FCMFEncoder(pretrained_path, num_imgs, num_roi, alpha)
        self.text_pooler = BertPooler()
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Linear(HIDDEN_SIZE * 2, num_labels) # Nhân đôi kích thước đầu vào do kết hợp CLS + Max Pooling
        # Chỉ khởi tạo các module MỚI (không khởi tạo encoder đã pre-trained)
        self._init_weights(self.text_pooler)
        self._init_weights(self.classifier)
        
    # Hàm khởi tạo trọng số chuẩn BERT
    def _init_weights(self, module):
        """Initialize the weights"""
        for m in module.modules():  # Duyệt tất cả sub-module
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def apply_custom_init(self, module):
        module.apply(self._init_weights)
    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors = None, token_type_ids=None, attention_mask=None, added_attention_mask=None):
        
        output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        if isinstance(output, tuple):
            sequence_output = output[0] # [Batch, 184, 768] (Gồm Text + Visual)
        else:
            sequence_output = output
        
        # 1. Lấy [CLS] Feature
        cls_output = self.text_pooler(sequence_output)
        
        # 2. Max Pooling
        # Lấy chiều dài thực tế của phần văn bản từ mask (thường là 170)
        text_len = attention_mask.shape[1]
        
        # Cắt sequence_output chỉ giữ lại phần Text (bỏ token visual ở đuôi đi)
        text_sequence_output = sequence_output[:, :text_len, :] # [Batch, 170, 768]
        
        # Gán giá trị rất thấp cho vị trí padding trước max pooling
        # Tạo mask: [Batch, 170, 768]
        mask_expanded = attention_mask[:, :text_len].unsqueeze(-1).expand(text_sequence_output.size()).float()
        text_sequence_output_masked = text_sequence_output.clone()
        text_sequence_output_masked[mask_expanded == 0] = float('-inf')
        
        # Max pooling over sequence dimension
        max_pooled_output, _ = torch.max(text_sequence_output_masked, dim=1) # [Batch, 768]

        # Kết hợp CLS + Max Pooling
        combined_output = torch.cat((cls_output, max_pooled_output), dim=1)
        
        pooled_output = self.dropout(combined_output)
        logits = self.classifier(pooled_output) 
        return logits   
        
        # # 1. Encoder trả về Full Sequence [Batch, Seq_Len, Hidden]
        # sequence_output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        # if isinstance(sequence_output, tuple): 
        #     sequence_output = sequence_output[0]

        # # 2. [THAY ĐỔI LỚN] MEAN POOLING thay vì lấy CLS
        # # Tạo mask để không tính trung bình vào các token padding
        # # attention_mask shape: [Batch, Seq_Len]
        
        # # Lấy độ dài của mask văn bản (170)
        # seq_len = attention_mask.shape[1] 
        
        # # Chỉ lấy phần output tương ứng với văn bản (bỏ phần đuôi ảnh đi)
        # # Lưu ý: Phần text này ĐÃ chứa thông tin ảnh nhờ cơ chế Attention.
        # text_output = sequence_output[:, :seq_len, :] # [Batch, 170, 768]

        # # 2. Mean Pooling (kích thước đã khớp 170 vs 170)
        # mask_expanded = attention_mask.unsqueeze(-1).expand(text_output.size()).float()
        
        # sum_embeddings = torch.sum(text_output * mask_expanded, 1)
        # sum_mask = mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # mean_pooled_output = sum_embeddings / sum_mask
        
        # # 3. Phân loại
        # pooled_output = self.dropout(mean_pooled_output)
        # logits = self.classifier(pooled_output)

        # return logits