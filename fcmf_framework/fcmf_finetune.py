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
        self.encoder = FCMFEncoder(pretrained_path, num_labels, num_imgs, num_roi, alpha)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)

    def forward(self, input_ids, visual_embeds_att, roi_embeds_att, roi_coors = None, token_type_ids=None, attention_mask=None, added_attention_mask=None):

        pooled_output = self.encoder(input_ids, visual_embeds_att, roi_embeds_att, roi_coors, token_type_ids, attention_mask, added_attention_mask)
        logits = self.classifier(pooled_output)
        return logits   