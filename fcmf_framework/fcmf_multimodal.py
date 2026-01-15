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
class FCMF(nn.Module):
    def __init__(self, pretrained_path, num_labels=4, num_imgs = 7, num_roi = 7, alpha=0.7):
        super(FCMF, self).__init__()
        self.encoder = FCMFEncoder(pretrained_path, num_imgs, num_roi, alpha)
        # self.text_pooler = BertPooler()
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)
        # self.apply_custom_init(self.text_pooler)
        self.apply_custom_init(self.classifier)

    # Hàm khởi tạo trọng số chuẩn BERT
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Dùng Normal distribution thay vì Uniform mặc định
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply_custom_init(self, module):
        module.apply(self._init_weights)
    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors = None, token_type_ids=None, attention_mask=None, added_attention_mask=None):
        
        # output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        # if isinstance(output, tuple):
        #     sequence_output = output[0] # get the last hidden states
        # else:
        #     sequence_output = output
        
        # pooled_output = self.text_pooler(sequence_output)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output) 
        # return logits   
        
        # 1. Encoder trả về Full Sequence [Batch, Seq_Len, Hidden]
        sequence_output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        if isinstance(sequence_output, tuple): 
            sequence_output = sequence_output[0]

        # 2. [THAY ĐỔI LỚN] MEAN POOLING thay vì lấy CLS
        # Tạo mask để không tính trung bình vào các token padding
        # attention_mask shape: [Batch, Seq_Len]
        
        # Lấy độ dài của mask văn bản (170)
        seq_len = attention_mask.shape[1] 
        
        # Chỉ lấy phần output tương ứng với văn bản (bỏ phần đuôi ảnh đi)
        # Lưu ý: Phần text này ĐÃ chứa thông tin ảnh nhờ cơ chế Attention.
        text_output = sequence_output[:, :seq_len, :] # [Batch, 170, 768]

        # 2. Mean Pooling (kích thước đã khớp 170 vs 170)
        mask_expanded = attention_mask.unsqueeze(-1).expand(text_output.size()).float()
        
        sum_embeddings = torch.sum(text_output * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        mean_pooled_output = sum_embeddings / sum_mask
        
        # 3. Phân loại
        pooled_output = self.dropout(mean_pooled_output)
        logits = self.classifier(pooled_output)

        return logits