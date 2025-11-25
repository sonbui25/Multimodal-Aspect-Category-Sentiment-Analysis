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
        self.text_pooler = BertPooler()
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)

    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors = None, token_type_ids=None, attention_mask=None, added_attention_mask=None):

        output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        pooled_output = self.text_pooler(output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 
        return logits   