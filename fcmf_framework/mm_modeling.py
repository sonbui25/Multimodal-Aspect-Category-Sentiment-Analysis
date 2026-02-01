import torch
import torch.nn as nn
import copy
import numpy as np
import math
import torch.nn.functional as F
from transformers import AutoModel
from torch.autograd import Variable 

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

HIDDEN_SIZE=768 # for base model, set to 1024 for large model
NUM_HIDDEN_LAYERS=12
NUM_ATTENTION_HEADS=12 
INTERMEDIATE_SIZE=3072
HIDDEN_ACT="gelu"
HIDDEN_DROPOUT_PROB=0.1
ATTENTION_PROBS_DROPOUT_PROB=0.1
MAX_POSITION_EMBEDDINGS=512
TYPE_VOCAB_SIZE=2
INITIALIZER_RANGE=0.02

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

# for NAACL visual attention part
class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        # learnable projection weights per head (initialized below)
        self.w_kx = nn.Parameter(torch.empty(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.empty(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, embed_dim)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.empty(hidden_dim * 2, 1))
        elif score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

        # init
        nn.init.xavier_uniform_(self.w_kx)
        nn.init.xavier_uniform_(self.w_qx)
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

        # store weights for visualization
        self.attention_weights = None


    def forward(self, k, q, memory_len=None):
        if k.dim() == 2:
            k = k.unsqueeze(1)
        if q.dim() == 2:
            q = q.unsqueeze(1)

        mb_size = k.size(0)
        k_len = k.size(1)
        q_len = q.size(1)
        device = k.device
        dtype = k.dtype

        # project per-head (repeat weights to match batch)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head * mb_size, k_len, self.embed_dim)
        w_k_rep = self.w_kx.repeat(mb_size, 1, 1).view(self.n_head * mb_size, self.embed_dim, self.hidden_dim).to(device=device, dtype=dtype)
        kx = torch.bmm(kx.to(dtype=dtype), w_k_rep)

        qx = q.repeat(self.n_head, 1, 1).view(self.n_head * mb_size, q_len, self.embed_dim)
        w_q_rep = self.w_qx.repeat(mb_size, 1, 1).view(self.n_head * mb_size, self.embed_dim, self.hidden_dim).to(device=device, dtype=dtype)
        qx = torch.bmm(qx.to(dtype=dtype), w_q_rep)

        if self.score_function == 'scaled_dot_product':
            kt = kx.transpose(1, 2)  # (nh*mb, hidden, k_len)
            qkt = torch.bmm(qx, kt)   # (nh*mb, q_len, k_len)
            # divide by sqrt(d_k) using float-safe conversion
            denom = math.sqrt(self.hidden_dim)
            score = qkt / denom
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.tanh(torch.bmm(qw, kt))
        else:
            raise RuntimeError('invalid score_function')

        # build mask (ensure shapes match score: (nh*mb, q_len, k_len))
        mask = None
        if memory_len is not None:
            if isinstance(memory_len, (list, tuple)):
                memory_len = torch.tensor(memory_len, device=device)
            if isinstance(memory_len, torch.Tensor) and memory_len.dim() == 1:
                # create per-batch mask and repeat for heads
                idx = torch.arange(k_len, device=device).unsqueeze(0)  # (1, k_len)
                mask_b = (idx < memory_len.unsqueeze(1)).float()      # (mb, k_len)
                mask = mask_b.unsqueeze(1).repeat(self.n_head, 1, 1)  # (nh*mb, 1, k_len) -> later broadcast
            elif isinstance(memory_len, torch.Tensor) and memory_len.dim() == 2:
                # casual_mask case
                casual_mask = torch.tril(torch.ones(q_len, k_len, device=device))
                mask = casual_mask.unsqueeze(0)  # shape (1, q_len, k_len)

        if mask is not None:
            # ensure mask broadcast to (nh*mb, q_len, k_len)
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.repeat(self.n_head * mb_size, 1, 1)
            score = score.masked_fill(mask == 0, float('-1e4'))

        score = F.softmax(score, dim=-1)
        self.attention_weights = score

        output = torch.bmm(score, kx)  # (nh*mb, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (mb, q_len, nh*hidden)
        output = self.proj(output)
        return output, score


# class SelfAttention(Attention): #1
#     '''q is a parameter'''

#     def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', q_len=1, dropout=0.1):
#         super(SelfAttention, self).__init__(embed_dim, hidden_dim, n_head, score_function, dropout)
#         self.q_len = q_len
#         self.q = nn.Parameter(torch.FloatTensor(q_len, embed_dim))

#     def forward(self, k, **kwargs):
#         mb_size = k.shape[0]
#         q = self.q.expand(mb_size, -1, -1)
#         return super(SelfAttention, self).forward(k, q)

class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class FCMFLayerNorm(nn.Module): 
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(FCMFLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module): 
    def __init__(self):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.attention_head_size = int(HIDDEN_SIZE / NUM_ATTENTION_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.key = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.value = nn.Linear(HIDDEN_SIZE, self.all_head_size)

        self.dropout = nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertCoAttention(nn.Module): 
    def __init__(self):
        super(BertCoAttention, self).__init__()

        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.attention_head_size = int(HIDDEN_SIZE / NUM_ATTENTION_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.key = nn.Linear(HIDDEN_SIZE, self.all_head_size)
        self.value = nn.Linear(HIDDEN_SIZE, self.all_head_size)

        self.dropout = nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module): 
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.LayerNorm = FCMFLayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module): 
    def __init__(self):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertCrossAttention(nn.Module): 
    def __init__(self):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention()
        self.output = BertSelfOutput()

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertIntermediate(nn.Module): # in BertLayer and BertCrossAttentionLayer
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.intermediate_act_fn = ACT2FN[HIDDEN_ACT]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.LayerNorm = FCMFLayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module): 
    def __init__(self):
        super(BertLayer, self).__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCrossAttentionLayer(nn.Module): 
    def __init__(self):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

# class BertEncoder(nn.Module): 
#     def __init__(self):
#         super(BertEncoder, self).__init__()
#         layer = BertLayer()
#         self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(NUM_HIDDEN_LAYERS)])

#     def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
#         all_encoder_layers = []
#         for layer_module in self.layer:
#             hidden_states = layer_module(hidden_states, attention_mask)
#             if output_all_encoded_layers:
#                 all_encoder_layers.append(hidden_states)
#         if not output_all_encoded_layers:
#             all_encoder_layers.append(hidden_states)
#         return all_encoder_layers

class MultimodalEncoder(nn.Module): 
    def __init__(self):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertCrossEncoder(nn.Module): 
    def __init__(self):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class BertText1Pooler(nn.Module): 
    def __init__(self):
        super(BertText1Pooler, self).__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token of text.
        first_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertPooler(nn.Module): 
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class FeatureExtractor(torch.nn.Module): 
  def __init__(self, pretrained_path):
    super(FeatureExtractor,self).__init__()
    self.cell = AutoModel.from_pretrained(pretrained_path,
                                          local_files_only=True,
                                          attn_implementation="eager")

  def forward(self, input_ids, token_type_ids, attention_mask):
    seq_out, pooled_out, enc_attentions = self.cell(input_ids = input_ids,
                                    token_type_ids = token_type_ids, 
                                    attention_mask = attention_mask,
                                    output_attentions=True)[:3]

    return seq_out, pooled_out, enc_attentions

class MultimodalDenoisingEncoder(nn.Module):
    #Module lọc nhiễu ảnh
    def __init__(self, alpha=0.7): 
        """
        Multimodal Denoising Encoder (MDE) với cơ chế Max-Pooling Fusion.
        """
        super(MultimodalDenoisingEncoder, self).__init__()
        self.alpha = alpha
        self.hidden_size = HIDDEN_SIZE
        
        # Tận dụng class Attention có sẵn cho bước Guidance
        # Lưu ý: Input ảnh vào đây là Key/Value, Text là Query
        self.guidance_attention = Attention(
            embed_dim=self.hidden_size, 
            hidden_dim=self.hidden_size // NUM_ATTENTION_HEADS,
            n_head=NUM_ATTENTION_HEADS, 
            score_function='scaled_dot_product', 
            dropout=0.1
        )
        
        # self.LayerNorm = FCMFLayerNorm(self.hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

    def forward(self, text_hidden_states, image_hidden_states):
        """
        Args:
            text_hidden_states: [Batch, Seq, Hidden]
            image_hidden_states: [Batch, N, Hidden]
        """
        B, N, H = image_hidden_states.shape

        
        # 1. Scoring
        
        text_query = text_hidden_states[:, 0, :].unsqueeze(1)
        dummy_len = [N] * B

        # Tính Score thô
        _, raw_scores = self.guidance_attention(image_hidden_states, text_query, dummy_len)
        # [B, 12, 1, 49] -> [B, 49]
        scores = raw_scores.view(B, NUM_ATTENTION_HEADS, 1, N).mean(dim=1).squeeze(1)

        
        # 2. Top-K Selection
        
        k_strong = max(1, int(N * self.alpha))

        m_weak = N - k_strong

        _, idx_strong = torch.topk(scores, k=k_strong, dim=1, largest=True)
        _, idx_weak = torch.topk(scores, k=m_weak, dim=1, largest=False)

        # Helper gather
        def gather(feat, idx):
            expand_idx = idx.unsqueeze(-1).expand(-1, -1, H)
            return torch.gather(feat, 1, expand_idx)

        v_strong = gather(image_hidden_states, idx_strong)
        v_weak = gather(image_hidden_states, idx_weak)

        
        # 3. Cosine Similarity
        
        v_strong_norm = F.normalize(v_strong, p=2, dim=-1)
        v_weak_norm = F.normalize(v_weak, p=2, dim=-1)
        similarity_matrix = torch.matmul(v_weak_norm, v_strong_norm.transpose(-1, -2))

        
        # 4. Theta & Assignment
        
        max_sim_vals, assignment_indices = torch.max(similarity_matrix, dim=-1)

        e_val = math.e
        exp_S = torch.exp(max_sim_vals)
        theta_weak = exp_S / (exp_S + e_val)

        
        # 5. Max-Pool Fusion
        
        # A. Assignment Mask
        assignment_mask = F.one_hot(assignment_indices, num_classes=k_strong).float()  # [B, M, K]

        # C. Max-Pooling
        v_weak_expanded = v_weak.unsqueeze(2)
        mask_expanded = assignment_mask.unsqueeze(-1)  # [B, M, K, 1]

        # Gán -inf cho những ô không thuộc nhóm
        vectors_to_pool = v_weak_expanded.masked_fill(mask_expanded == 0, -1e4)

        attended_weak_patches, _ = torch.max(vectors_to_pool, dim=1)

        # D. Xử lý ô trống
        has_valid_child = (torch.sum(assignment_mask, dim=1, keepdim=True) > 0)
        attended_weak_patches = attended_weak_patches.masked_fill(~has_valid_child.transpose(1, 2), 0.0)

        # E. Max-Pool Theta
        theta_map = theta_weak.unsqueeze(-1) * assignment_mask
        theta_map = theta_map.masked_fill(assignment_mask == 0, -1e4)
        theta_strong, _ = torch.max(theta_map, dim=1)
        theta_strong = theta_strong.masked_fill(theta_strong == -1e4, 0.0).unsqueeze(-1)

        # F. Update
        v_strong_updated = (1 - theta_strong) * v_strong + theta_strong * attended_weak_patches

        
        # 6. Return
        
        return v_strong_updated

    # Transformer Decoder related modules
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(HIDDEN_SIZE, ffn_num_hiddens)
        self.act = ACT2FN[HIDDEN_ACT]
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, x):
        return self.dense2(self.act(self.dense1(x)))
class AddNorm(nn.Module):
    "The residual connection followed by layer normalization."
    def __init__(self, norm_shape,  dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = FCMFLayerNorm(norm_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
class TransformerDecoderBlock(nn.Module):
    # The i-th block of the transformer decoder
    def __init__(self, i):
        super(TransformerDecoderBlock, self).__init__()
        self.i = i
        self.attention1 = Attention(HIDDEN_SIZE, HIDDEN_SIZE // NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS, 'scaled_dot_product', ATTENTION_PROBS_DROPOUT_PROB)
        self.addnorm1 = AddNorm(HIDDEN_SIZE, ATTENTION_PROBS_DROPOUT_PROB)
        self.attention2 = Attention(HIDDEN_SIZE, HIDDEN_SIZE // NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS, 'scaled_dot_product', ATTENTION_PROBS_DROPOUT_PROB)
        self.addnorm2 = AddNorm(HIDDEN_SIZE, ATTENTION_PROBS_DROPOUT_PROB)
        self.ffn = PositionWiseFFN(HIDDEN_SIZE, HIDDEN_SIZE)
        self.add_norm3 = AddNorm(HIDDEN_SIZE, ATTENTION_PROBS_DROPOUT_PROB)
    def forward(self, X, state, enc_attention_mask=None, is_train=True):
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
            state[2][self.i] = key_values
            
        if is_train:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        X2, _ = self.attention1(X, X, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        # Encoder-decoder attention (Cross-Attention)
        # Ưu tiên dùng enc_attention_mask nếu có.
        # Nếu không (fallback), dùng enc_valid_lens cũ.
        cross_mask = enc_attention_mask if enc_attention_mask is not None else enc_valid_lens
        
        # Y là Query (Decoder), enc_outputs là Key/Value (Encoder)
        Y2, _ = self.attention2(enc_outputs, Y, cross_mask)
        
        Z = self.addnorm2(Y, Y2)
        return self.add_norm3(Z, self.ffn(Z)), state
    
class PositionalEncoding(nn.Module): 
    """Positional encoding."""
    def __init__(self): 
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB)
        # Create a long enough P using MAX_POSITION_EMBEDDINGS
        P = torch.zeros((1, MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE))
        X = torch.arange(MAX_POSITION_EMBEDDINGS, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, HIDDEN_SIZE, 2, dtype=torch.float32) / HIDDEN_SIZE)
        P[:, :, 0::2] = torch.sin(X)
        P[:, :, 1::2] = torch.cos(X)
        self.register_buffer('P', P)

    def forward(self, X):
        pe = self.P[:, :X.size(1), :].to(device=X.device).type_as(X)
    
        X = X + pe
        return self.dropout(X)
class IAOGDecoder(nn.Module):
    def __init__(self, vocab_size):
        super(IAOGDecoder, self).__init__()
        self.num_hiddens = HIDDEN_SIZE
        self.num_blks = NUM_HIDDEN_LAYERS
        self.embedding = nn.Embedding(vocab_size, self.num_hiddens)
        self.pos_encoding = PositionalEncoding()
        self.blks = nn.Sequential()
        for i in range(NUM_HIDDEN_LAYERS):
            self.blks.add_module('block' + str(i), TransformerDecoderBlock(i))
        self.dense = nn.Linear(self.num_hiddens, vocab_size)
        self.dense.weight = self.embedding.weight
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state, enc_attention_mask=None, is_train=True):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        
        # XỬ LÝ MASK
        for i, blk in enumerate(self.blks):
            # Truyền mask xuống block
            X, state = blk(X, state, enc_attention_mask=enc_attention_mask, is_train=is_train)
            
            self._attention_weights[0][i] = blk.attention1.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention_weights
            
        return self.dense(X)

    @property
    def attention_weights(self):
        return self._attention_weights

        
        