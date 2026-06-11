"""
自包含的教师情感识别模型架构 — 完全解耦版本。

本文件从主项目中复制了模型推理所需的全部代码，不依赖任何 src/ 下的模块。
来源文件:
  - src/model.py         (模型架构)
  - src/sequence_utils.py (序列工具)
  - src/context_aware.py  (教学情境感知模块)

模型: PromptModel (MPLMM + 三个创新点)
输入: text(768-d), audio(1024-d), vision(28-d)
输出: 4 类教师情感 (enthusiastic / calm / negative / anxious)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ============================================================
# 序列工具 (来自 src/sequence_utils.py)
# ============================================================

def make_padding_mask(seq_lengths, max_len, device):
    """Build a padding mask with shape ``(batch, max_len)``; True means padding."""
    if seq_lengths is None:
        return None
    lengths = seq_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=max_len)
    steps = torch.arange(max_len, device=device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


# ============================================================
# 多头注意力机制
# ============================================================

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


# ============================================================
# 正弦位置编码
# ============================================================

def make_positions(tensor, padding_idx, left_pad):
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.device
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name,
            getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        setattr(make_positions, buf_name,
                torch.arange(padding_idx + 1, max_pos).type_as(tensor))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.device
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(
            0, positions.contiguous().view(-1)
        ).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        return int(1e5)


# ============================================================
# Transformer 编码器
# ============================================================

def _fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def _buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(_fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]


def _Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def _LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1,
                 relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout)
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = _Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = _Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([_LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, key_padding_mask=None, return_attn=False):
        attn_weights = None
        residual = x
        x = self._maybe_layer_norm(0, x, before=True)
        mask = _buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, attn_weights = self.self_attn(
                query=x, key=x, value=x, attn_mask=mask, key_padding_mask=key_padding_mask)
        else:
            x_k = self._maybe_layer_norm(0, x_k, before=True)
            x_v = self._maybe_layer_norm(0, x_v, before=True)
            x, attn_weights = self.self_attn(
                query=x, key=x_k, value=x_v, attn_mask=mask, key_padding_mask=key_padding_mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(0, x, after=True)

        residual = x
        x = self._maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(1, x, after=True)

        if return_attn:
            return x, attn_weights
        return x

    def _maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0,
                 relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0,
                 attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerEncoderLayer(
                embed_dim, num_heads=num_heads, attn_dropout=attn_dropout,
                relu_dropout=relu_dropout, res_dropout=res_dropout, attn_mask=attn_mask))

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = _LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None, key_padding_mask=None, return_attn=False):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        intermediates = [x]
        all_attn_weights = [] if return_attn else None

        for i, layer in enumerate(self.layers):
            if x_in_k is not None and x_in_v is not None:
                if return_attn:
                    x, attn_weights = layer(x, x_k, x_v, key_padding_mask=key_padding_mask, return_attn=True)
                    all_attn_weights.append(attn_weights)
                else:
                    x = layer(x, x_k, x_v, key_padding_mask=key_padding_mask)
            else:
                if return_attn:
                    x, attn_weights = layer(x, key_padding_mask=key_padding_mask, return_attn=True)
                    all_attn_weights.append(attn_weights)
                else:
                    x = layer(x, key_padding_mask=key_padding_mask)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        if return_attn:
            return x, {'attn_weights': all_attn_weights, 'intermediates': intermediates}
        return x

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


# ============================================================
# MLP 层
# ============================================================

class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


# ============================================================
# 创新点1: 时序 Prompt 模块
# ============================================================

class TemporalPromptInjection(nn.Module):
    def __init__(self, embed_dim, max_seq_len=30):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.temporal_pe = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.constant_(gate_linear.bias, -2.0)
        self.gate = nn.Sequential(gate_linear, nn.Sigmoid())

    def forward(self, x):
        seq_len = x.size(0)
        t_pe = self.temporal_pe[:seq_len].unsqueeze(1).expand(-1, x.size(1), -1)
        gate_val = self.gate(torch.cat([x, t_pe], dim=-1))
        return x + gate_val * t_pe


class TemporalAggregation(nn.Module):
    """时序聚合层 — 与 checkpoint 匹配的简化版本 (仅 attn_pool)。"""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.Tanh(), nn.Linear(embed_dim // 4, 1))

    def forward(self, x, padding_mask=None, actual_seq_len=None):
        if actual_seq_len is not None:
            x_valid = x[:actual_seq_len]
        else:
            x_valid = x

        x_t = x_valid.transpose(0, 1)  # (batch, seq, dim)
        scores = self.attn_pool(x_t)    # (batch, seq, 1)
        if padding_mask is not None:
            valid_mask = padding_mask[:, :actual_seq_len] if actual_seq_len else padding_mask
            scores = scores.masked_fill(valid_mask.unsqueeze(-1), float('-inf'))
        w = F.softmax(scores, dim=1)    # (batch, seq, 1)
        return (x_t * w).sum(dim=1)     # (batch, dim)


# ============================================================
# 创新点2: 教学情境感知模块
# ============================================================

class TimePositionEncoding(nn.Module):
    NUM_FREQ = 4
    OUTPUT_DIM = 2 + 2 * NUM_FREQ  # = 10

    @staticmethod
    def get_position_features(seq_lengths, max_seq_len):
        batch_size = seq_lengths.size(0)
        device = seq_lengths.device
        num_freq = TimePositionEncoding.NUM_FREQ

        positions = torch.arange(max_seq_len, device=device).float()
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        lengths = seq_lengths.float().clamp(min=1).unsqueeze(1)
        pos_norm = positions / (lengths - 1).clamp(min=1)
        pos_norm = pos_norm.clamp(0, 1)

        mask = (positions < seq_lengths.unsqueeze(1)).float()

        feat_basic = torch.stack([pos_norm, 1 - pos_norm], dim=-1)

        freqs = torch.arange(1, num_freq + 1, device=device).float()
        angles = pos_norm.unsqueeze(-1) * freqs * math.pi
        feat_sin = torch.sin(angles)
        feat_cos = torch.cos(angles)

        time_pos = torch.cat([feat_basic, feat_sin, feat_cos], dim=-1)
        time_pos = time_pos * mask.unsqueeze(-1)
        return time_pos

    @staticmethod
    def get_segment_summary_features(seq_lengths, max_seq_len):
        batch_size = seq_lengths.size(0)
        device = seq_lengths.device
        dtype = torch.float32
        num_freq = TimePositionEncoding.NUM_FREQ

        max_len = max(int(max_seq_len), 1)
        lengths = seq_lengths.to(device=device, dtype=dtype).clamp(min=1, max=max_len)
        denom = float(max(max_len - 1, 1))

        center_norm = ((lengths - 1.0) / 2.0) / denom
        length_ratio = lengths / float(max_len)

        feat_basic = torch.stack([center_norm, length_ratio], dim=-1)

        freqs = torch.arange(1, num_freq + 1, device=device, dtype=dtype)
        angles = center_norm.unsqueeze(-1) * freqs * math.pi
        feat_sin = torch.sin(angles)
        feat_cos = torch.cos(angles)

        summary = torch.cat([feat_basic, feat_sin, feat_cos], dim=-1)
        return summary.view(batch_size, TimePositionEncoding.OUTPUT_DIM)

    def forward(self, x):
        # 不会被直接调用，使用静态方法
        raise NotImplementedError("Use static methods instead")


class ContextConditionedModalityWeight(nn.Module):
    """情境条件化的模态权重生成 — 与 checkpoint 匹配的版本 (weight_net)。"""
    def __init__(self, num_activities=4, hidden_dim=64, feat_dim=2):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(num_activities, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

    def forward(self, activity_probs=None, feat_repr=None, batch_size=None):
        if activity_probs is not None:
            raw_weights = self.weight_net(activity_probs)
        elif feat_repr is not None:
            raw_weights = self.weight_net(feat_repr)
        else:
            bs = batch_size or 4
            return torch.ones(bs, 3, device='cpu') / 3.0
        return F.softmax(raw_weights, dim=-1)


class AdaptiveMissingModalityHandler(nn.Module):
    MISSING_WEIGHT_MASK = torch.tensor([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0],
    ], dtype=torch.float32)

    def __init__(self, combined_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(combined_dim + 3, combined_dim), nn.Sigmoid())

    def forward(self, last_hs, modality_weights, missing_mod):
        mask = self.MISSING_WEIGHT_MASK.to(
            device=modality_weights.device, dtype=modality_weights.dtype)[missing_mod]
        adjusted_weights = modality_weights * (1.0 - mask)
        weight_sum = adjusted_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adjusted_weights = adjusted_weights / weight_sum
        x = torch.cat([last_hs, adjusted_weights], dim=-1)
        gate_val = self.gate(x)
        return last_hs * gate_val


# ============================================================
# 教学活动分类器 (checkpoint 中的 activity_classifier)
# ============================================================

class TeachingActivityClassifier(nn.Module):
    """教学活动分类器 — 从文本+音频聚合表示预测教学活动。"""
    def __init__(self, text_dim, audio_dim, hidden_dim=256, num_activities=4):
        super().__init__()
        self.num_activities = num_activities
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_activities),
        )

    def forward(self, text_feat, audio_feat):
        x = torch.cat([text_feat, audio_feat], dim=-1)
        return self.classifier(x)


# ============================================================
# PromptModel: 带提示学习的主模型 (与 checkpoint 匹配)
# ============================================================

class PromptModel(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v)
        self.d_l, self.d_a, self.d_v = (
            hyp_params.proj_dim, hyp_params.proj_dim, hyp_params.proj_dim)
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = hyp_params.prompt_dim
        self.llen, self.alen, self.vlen = hyp_params.seq_len

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = hyp_params.output_dim

        # 消融开关
        self.use_generative_prompt = getattr(hyp_params, 'use_generative_prompt', True)
        self.use_modality_signal = getattr(hyp_params, 'use_modality_signal', True)
        self.use_missing_type_prompt = getattr(hyp_params, 'use_missing_type_prompt', True)

        # 生成提示
        self.generative_prompt = nn.Parameter(torch.zeros(3, self.prompt_dim, self.prompt_length))

        # 跨模态 MLP (6个)
        self.l2a = MLPLayer(self.orig_d_l, self.prompt_dim)
        self.l2v = MLPLayer(self.orig_d_l, self.prompt_dim)
        self.v2a = MLPLayer(self.orig_d_v, self.prompt_dim)
        self.v2l = MLPLayer(self.orig_d_v, self.prompt_dim)
        self.a2v = MLPLayer(self.orig_d_a, self.prompt_dim)
        self.a2l = MLPLayer(self.orig_d_a, self.prompt_dim)

        # 模态生成 MLP (9个)
        self.l_ap = MLPLayer(self.prompt_length + self.alen, self.llen, True)
        self.l_vp = MLPLayer(self.prompt_length + self.vlen, self.llen, True)
        self.l_avp = MLPLayer(self.prompt_length + self.alen + self.vlen, self.llen, True)
        self.a_lp = MLPLayer(self.prompt_length + self.llen, self.alen, True)
        self.a_vp = MLPLayer(self.prompt_length + self.vlen, self.alen, True)
        self.a_lvp = MLPLayer(self.prompt_length + self.llen + self.vlen, self.alen, True)
        self.v_ap = MLPLayer(self.prompt_length + self.alen, self.vlen, True)
        self.v_lp = MLPLayer(self.prompt_length + self.llen, self.vlen, True)
        self.v_alp = MLPLayer(self.prompt_length + self.alen + self.llen, self.vlen, True)

        # 投影层
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 模态信号提示 (6个)
        self.promptl_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.promptl_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # 缺失类型提示
        self.missing_type_prompt = nn.Parameter(torch.zeros(3, self.prompt_length, self.prompt_dim))
        self.m_a = nn.Parameter(torch.zeros(self.alen, 2 * self.prompt_dim))
        self.m_v = nn.Parameter(torch.zeros(self.vlen, 2 * self.prompt_dim))
        self.m_l = nn.Parameter(torch.zeros(self.llen, 2 * self.prompt_dim))

        # 跨模态注意力
        self.trans_l_with_a = self._get_network(self_type="la")
        self.trans_l_with_v = self._get_network(self_type="lv")
        self.trans_a_with_l = self._get_network(self_type="al")
        self.trans_a_with_v = self._get_network(self_type="av")
        self.trans_v_with_l = self._get_network(self_type="vl")
        self.trans_v_with_a = self._get_network(self_type="va")

        # 自注意力
        self.trans_l_mem = self._get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self._get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self._get_network(self_type="v_mem", layers=3)

        # 输出投影层
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # 教学活动分类器 (checkpoint 中存在)
        self.activity_classifier = TeachingActivityClassifier(2 * self.d_l, 2 * self.d_a)

        # 创新点1: 时序 Prompt
        self.use_temporal = getattr(hyp_params, 'use_temporal', False)
        if self.use_temporal:
            max_seq_len = getattr(hyp_params, 'max_seq_len', 30)
            self.temporal_inject_l = TemporalPromptInjection(self.d_l, max_seq_len=max_seq_len)
            self.temporal_inject_a = TemporalPromptInjection(self.d_a, max_seq_len=max_seq_len)
            self.temporal_inject_v = TemporalPromptInjection(self.d_v, max_seq_len=max_seq_len)
            self.temporal_agg_l = TemporalAggregation(2 * self.d_l)
            self.temporal_agg_a = TemporalAggregation(2 * self.d_a)
            self.temporal_agg_v = TemporalAggregation(2 * self.d_v)

        # 创新点2: 教学情境感知
        self.use_context = getattr(hyp_params, 'use_context', False)
        if self.use_context:
            context_feat_dim = TimePositionEncoding.OUTPUT_DIM + combined_dim
            self.context_weight = ContextConditionedModalityWeight(
                num_activities=4, hidden_dim=64, feat_dim=context_feat_dim)
            self.adaptive_missing = AdaptiveMissingModalityHandler(combined_dim=combined_dim)

        # 创新点3: 跨域 Prompt 迁移
        # 注意: 跨域模块需要额外依赖 (src/cross_domain.py)，
        # full_model 配置未启用此功能。如需支持，请将 cross_domain.py 也复制到本目录。
        self.use_cross_domain = getattr(hyp_params, 'use_cross_domain', False)
        if self.use_cross_domain:
            raise NotImplementedError(
                "跨域 Prompt 迁移模块 (use_cross_domain=True) 需要额外依赖。"
                "当前解耦版本不支持此功能。请将 src/cross_domain.py 复制到 core/ 目录。"
            )

    def _get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {self_type}")

        return TransformerEncoder(
            embed_dim=embed_dim, num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout, relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout, embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask)

    def _get_complete_data_batched(self, x_l, x_a, x_v, missing_mod, domain_prompts_batch=None):
        batch_size = x_l.size(0)
        device = x_l.device
        seq_len = x_l.size(2)

        use_gp = self.use_generative_prompt
        use_sig = self.use_modality_signal

        a2l_out = self.a2l(x_a)
        v2l_out = self.v2l(x_v)
        l2a_out = self.l2a(x_l)
        v2a_out = self.v2a(x_v)
        l2v_out = self.l2v(x_l)
        a2v_out = self.a2v(x_a)

        gp_l = self.generative_prompt[0]
        gp_a = self.generative_prompt[1]
        gp_v = self.generative_prompt[2]

        out_l = torch.empty(batch_size, self.prompt_dim, seq_len, device=device)
        out_a = torch.empty_like(out_l)
        out_v = torch.empty_like(out_l)

        sig_lm = self.promptl_m if use_sig else 0
        sig_lnm = self.promptl_nm if use_sig else 0
        sig_am = self.prompta_m if use_sig else 0
        sig_anm = self.prompta_nm if use_sig else 0
        sig_vm = self.promptv_m if use_sig else 0
        sig_vnm = self.promptv_nm if use_sig else 0

        def make_gp(base_gp, B, idx):
            if not use_gp:
                return torch.zeros(B, self.prompt_dim, self.prompt_length, device=device)
            g = base_gp.unsqueeze(0).expand(B, -1, -1)
            if domain_prompts_batch is not None:
                dp = domain_prompts_batch[idx].unsqueeze(-1).expand(-1, -1, self.prompt_length)
                g = g + dp
            return g

        for mode in range(7):
            idx = torch.where(missing_mod == mode)[0]
            if idx.numel() == 0:
                continue
            B = idx.size(0)

            if mode == 0:
                g = make_gp(gp_l, B, idx)
                cat = torch.cat([g, a2l_out[idx], v2l_out[idx]], dim=2)
                out_l[idx] = self.l_avp(cat.transpose(1, 2)).transpose(1, 2) + sig_lm
                out_a[idx] = self.proj_a(x_a[idx]) + sig_anm
                out_v[idx] = self.proj_v(x_v[idx]) + sig_vnm
            elif mode == 1:
                g = make_gp(gp_a, B, idx)
                cat = torch.cat([g, l2a_out[idx], v2a_out[idx]], dim=2)
                out_a[idx] = self.a_lvp(cat.transpose(1, 2)).transpose(1, 2) + sig_am
                out_v[idx] = self.proj_v(x_v[idx]) + sig_vnm
                out_l[idx] = self.proj_l(x_l[idx]) + sig_lnm
            elif mode == 2:
                g = make_gp(gp_v, B, idx)
                cat = torch.cat([g, l2v_out[idx], a2v_out[idx]], dim=2)
                out_v[idx] = self.v_alp(cat.transpose(1, 2)).transpose(1, 2) + sig_vm
                out_l[idx] = self.proj_l(x_l[idx]) + sig_lnm
                out_a[idx] = self.proj_a(x_a[idx]) + sig_anm
            elif mode == 3:
                gl = make_gp(gp_l, B, idx)
                ga = make_gp(gp_a, B, idx)
                cat_l = torch.cat([gl, v2l_out[idx]], dim=2)
                cat_a = torch.cat([ga, v2a_out[idx]], dim=2)
                out_l[idx] = self.l_vp(cat_l.transpose(1, 2)).transpose(1, 2) + sig_lm
                out_a[idx] = self.a_vp(cat_a.transpose(1, 2)).transpose(1, 2) + sig_am
                out_v[idx] = self.proj_v(x_v[idx]) + sig_vnm
            elif mode == 4:
                gl = make_gp(gp_l, B, idx)
                gv = make_gp(gp_v, B, idx)
                cat_l = torch.cat([gl, a2l_out[idx]], dim=2)
                cat_v = torch.cat([gv, a2v_out[idx]], dim=2)
                out_l[idx] = self.l_ap(cat_l.transpose(1, 2)).transpose(1, 2) + sig_lm
                out_v[idx] = self.v_ap(cat_v.transpose(1, 2)).transpose(1, 2) + sig_vm
                out_a[idx] = self.proj_a(x_a[idx]) + sig_anm
            elif mode == 5:
                ga = make_gp(gp_a, B, idx)
                gv = make_gp(gp_v, B, idx)
                cat_a = torch.cat([ga, l2a_out[idx]], dim=2)
                cat_v = torch.cat([gv, l2v_out[idx]], dim=2)
                out_a[idx] = self.a_lp(cat_a.transpose(1, 2)).transpose(1, 2) + sig_am
                out_v[idx] = self.v_lp(cat_v.transpose(1, 2)).transpose(1, 2) + sig_vm
                out_l[idx] = self.proj_l(x_l[idx]) + sig_lnm
            else:
                out_a[idx] = self.proj_a(x_a[idx]) + sig_anm
                out_l[idx] = self.proj_l(x_l[idx]) + sig_lnm
                out_v[idx] = self.proj_v(x_v[idx]) + sig_vnm

        return out_l, out_a, out_v

    def _get_proj_matrix(self):
        a_v_l = (self.prompta_nm @ self.m_a + self.promptv_nm @ self.m_v + self.promptl_nm @ self.m_l).unsqueeze(0)
        am_v_l = (self.prompta_m @ self.m_a + self.promptv_nm @ self.m_v + self.promptl_nm @ self.m_l).unsqueeze(0)
        a_vm_l = (self.prompta_nm @ self.m_a + self.promptv_m @ self.m_v + self.promptl_nm @ self.m_l).unsqueeze(0)
        a_v_lm = (self.prompta_nm @ self.m_a + self.promptv_nm @ self.m_v + self.promptl_m @ self.m_l).unsqueeze(0)
        am_vm_l = (self.prompta_m @ self.m_a + self.promptv_m @ self.m_v + self.promptl_nm @ self.m_l).unsqueeze(0)
        am_v_lm = (self.prompta_m @ self.m_a + self.promptv_nm @ self.m_v + self.promptl_m @ self.m_l).unsqueeze(0)
        a_vm_lm = (self.prompta_nm @ self.m_a + self.promptv_m @ self.m_v + self.promptl_m @ self.m_l).unsqueeze(0)
        self.mp = torch.cat([a_v_lm, am_v_l, a_vm_l, am_v_lm, a_vm_lm, am_vm_l, a_v_l], dim=0)

    @staticmethod
    def _make_padding_mask(seq_lengths, max_len, device):
        if seq_lengths is None:
            return None
        lengths = seq_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=max_len)
        steps = torch.arange(max_len, device=device).unsqueeze(0)
        return steps >= lengths.unsqueeze(1)

    @staticmethod
    def _make_memory_padding_mask(query_padding, prompt_length, batch_size, device):
        if query_padding is None:
            return None
        prompt_padding = torch.zeros(batch_size, prompt_length, dtype=torch.bool, device=device)
        return torch.cat([query_padding, prompt_padding], dim=1)

    @staticmethod
    def _masked_time_mean(x, seq_lengths):
        if seq_lengths is None:
            return x.mean(dim=-1)
        max_len = x.size(-1)
        lengths = seq_lengths.to(device=x.device, dtype=x.dtype).clamp(min=1, max=max_len)
        mask = (torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(torch.long).unsqueeze(1)).to(dtype=x.dtype)
        return (x * mask.unsqueeze(1)).sum(dim=-1) / lengths.unsqueeze(1)

    def _compute_context_weights(self, last_hs, last_h_l, last_h_a, last_h_v,
                                  seq_lengths, missing_mod, max_seq_len=None):
        batch_size = last_h_l.size(0)
        device = last_h_l.device
        seq_len = max_seq_len or self.llen

        # 使用教学活动分类器生成活动概率
        activity_logits = self.activity_classifier(last_h_l, last_h_a)
        activity_probs = F.softmax(activity_logits, dim=-1)

        # 情境条件化的模态权重 (基于活动概率)
        modality_weights = self.context_weight(activity_probs=activity_probs)

        # 自适应缺失模态处理
        last_hs = self.adaptive_missing(last_hs, modality_weights, missing_mod)

        return last_hs, modality_weights, activity_logits

    def forward(self, x_l, x_a, x_v, missing_mod, domain_labels=None, seq_lengths=None):
        if x_l.dim() == 2:
            x_l = x_l.unsqueeze(1)
        if x_a.dim() == 2:
            x_a = x_a.unsqueeze(1)
        if x_v.dim() == 2:
            x_v = x_v.unsqueeze(1)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        domain_prompts_batch = None
        if self.use_cross_domain and domain_labels is not None:
            feat_repr = torch.cat([
                self._masked_time_mean(x_l, seq_lengths),
                self._masked_time_mean(x_a, seq_lengths)], dim=-1)
            domain_prompts_batch, _ = self.domain_prompt_pool(feat_repr, domain_labels)

        xx_l, xx_a, xx_v = self._get_complete_data_batched(x_l, x_a, x_v, missing_mod, domain_prompts_batch)

        proj_x_a = xx_a.permute(2, 0, 1)
        proj_x_v = xx_v.permute(2, 0, 1)
        proj_x_l = xx_l.permute(2, 0, 1)
        batch_size = proj_x_l.size(1)

        if seq_lengths is not None:
            padding_l = self._make_padding_mask(seq_lengths, proj_x_l.size(0), proj_x_l.device)
            padding_a = self._make_padding_mask(seq_lengths, proj_x_a.size(0), proj_x_a.device)
            padding_v = self._make_padding_mask(seq_lengths, proj_x_v.size(0), proj_x_v.device)
        else:
            padding_l = padding_a = padding_v = None

        if self.use_temporal:
            proj_x_l = self.temporal_inject_l(proj_x_l)
            proj_x_a = self.temporal_inject_a(proj_x_a)
            proj_x_v = self.temporal_inject_v(proj_x_v)

        self._get_proj_matrix()

        if self.use_missing_type_prompt:
            mp_selected = self.mp[missing_mod].unsqueeze(1)
            batch_prompt = torch.matmul(
                self.missing_type_prompt.unsqueeze(0), mp_selected).transpose(0, 1)
        else:
            batch_prompt = torch.zeros(3, batch_size, self.prompt_length, 2 * self.d_l, device=x_l.device)

        # 文本模态融合
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a, key_padding_mask=padding_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v, key_padding_mask=padding_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = torch.cat([h_ls, batch_prompt[0].transpose(0, 1)], dim=0)
        mem_padding_l = self._make_memory_padding_mask(padding_l, self.prompt_length, batch_size, h_ls.device)
        h_ls = self.trans_l_mem(h_ls, key_padding_mask=mem_padding_l)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        if self.use_temporal:
            last_h_l = self.temporal_agg_l(h_ls, padding_mask=mem_padding_l, actual_seq_len=proj_x_l.size(0))
        else:
            last_h_l = h_ls[-1]

        # 音频模态融合
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l, key_padding_mask=padding_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, key_padding_mask=padding_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = torch.cat([h_as, batch_prompt[1].transpose(0, 1)], dim=0)
        mem_padding_a = self._make_memory_padding_mask(padding_a, self.prompt_length, batch_size, h_as.device)
        h_as = self.trans_a_mem(h_as, key_padding_mask=mem_padding_a)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        if self.use_temporal:
            last_h_a = self.temporal_agg_a(h_as, padding_mask=mem_padding_a, actual_seq_len=proj_x_a.size(0))
        else:
            last_h_a = h_as[-1]

        # 视觉模态融合
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l, key_padding_mask=padding_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a, key_padding_mask=padding_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = torch.cat([h_vs, batch_prompt[2].transpose(0, 1)], dim=0)
        mem_padding_v = self._make_memory_padding_mask(padding_v, self.prompt_length, batch_size, h_vs.device)
        h_vs = self.trans_v_mem(h_vs, key_padding_mask=mem_padding_v)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        if self.use_temporal:
            last_h_v = self.temporal_agg_v(h_vs, padding_mask=mem_padding_v, actual_seq_len=proj_x_v.size(0))
        else:
            last_h_v = h_vs[-1]

        # 直接拼接三模态聚合表示
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # 教学情境感知
        activity_logits = None
        if self.use_context:
            last_hs, _, activity_logits = self._compute_context_weights(
                last_hs, last_h_l, last_h_a, last_h_v,
                seq_lengths, missing_mod, max_seq_len=proj_x_l.size(0))

        # 残差投影块
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        # 跨域 Prompt 迁移
        domain_pred = None
        if self.use_cross_domain and domain_labels is not None:
            domain_repr = self.domain_prompt_pool.get_prompt_representation(domain_labels)
            domain_pred = self.domain_discriminator(domain_repr)

        if domain_pred is not None:
            return output, activity_logits, domain_pred
        return output, activity_logits, None
