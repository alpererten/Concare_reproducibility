import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "discretizer_config.json"


def _load_value_channel_mapping(config_path: Path):
    """
    Returns (channel_names, value_to_channel_index_list).
    """
    try:
        with config_path.open("r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        warnings.warn(f"Discretizer config not found at {config_path}. Falling back to uniform channel mapping.")
        return [], []

    channel_names = cfg["id_to_channel"]
    mapping = []
    for idx, ch in enumerate(channel_names):
        if cfg["is_categorical_channel"][ch]:
            width = len(cfg["possible_values"][ch])
        else:
            width = 1
        mapping.extend([idx] * width)
    return channel_names, mapping


class MissingAwareTemporalAttention(nn.Module):
    """
    ConCare time-aware attention augmented with SMART-style mask biasing.
    Observed timesteps receive a larger additive bias than missing ones.
    """
    def __init__(self, hidden_dim, attention_hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_hidden_dim = attention_hidden_dim

        self.Wt = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        self.Wx = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        self.rate = nn.Parameter(torch.zeros(1) + 0.8)
        self.obs_bias = nn.Parameter(torch.tensor(0.5))
        self.miss_bias = nn.Parameter(torch.tensor(0.0))

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, H, mask):
        """
        Args:
            H: [B, T, hidden_dim] GRU outputs for a single feature dimension.
            mask: [B, T] float/bool mask (1 if observed). If None, treated as all ones.
        Returns:
            context: [B, hidden_dim]
            attn:    [B, T] attention weights over timesteps
        """
        B, T, _ = H.size()
        device = H.device

        # Query/key projections identical to original ConCare attention
        q = torch.matmul(H[:, -1, :], self.Wt).unsqueeze(1)  # [B, 1, attn_dim]
        k = torch.matmul(H, self.Wx)                         # [B, T, attn_dim]
        dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze(1)  # [B, T]

        # Time indices from most recent back
        b_time = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1) + 1.0
        denom = self.sigmoid(self.rate) * (torch.log(2.72 + (1 - self.sigmoid(dot_product))) * b_time)
        e = self.relu(self.sigmoid(dot_product) / denom)

        if mask is None:
            mask = torch.ones(B, T, device=device)
        else:
            if mask.dim() != 2 or mask.size(1) != T:
                raise ValueError("Mask must be [B, T] to match hidden states.")
            mask = mask.to(device=device, dtype=torch.float32)

        bias = torch.where(mask > 0.5, self.obs_bias, self.miss_bias)
        e = e + bias

        a = self.softmax(e)
        v = torch.bmm(a.unsqueeze(1), H).squeeze(1)
        return v, a


class SingleAttentionPerFeatureNew(nn.Module):
    """
    Time-aware attention per feature using the authors' 'new' formulation.
    Matches their rate-controlled nonlinear time decay.
    """
    def __init__(self, hidden_dim, attention_hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_hidden_dim = attention_hidden_dim

        # Query from last timestep, Key from all timesteps
        self.Wt = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        self.Wx = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        # Learned rate that controls the time decay strength
        self.rate = nn.Parameter(torch.zeros(1) + 0.8)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, H):
        """
        H: [B, T, hidden_dim] from a feature-specific GRU
        Returns: [B, hidden_dim], attention weights [B, T]
        """
        B, T, _ = H.size()
        device = H.device

        # q: [B, 1, attn_hidden], k: [B, T, attn_hidden]
        q = torch.matmul(H[:, -1, :], self.Wt).unsqueeze(1)
        k = torch.matmul(H, self.Wx)

        # Dot attention scores across time
        # dot_product: [B, T]
        dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze(1)

        # Time indices from most recent back
        # b_time: [B, T]
        b_time = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1) + 1.0

        # Authors' nonlinear time-aware normalization
        # denominator: [B, T]
        denom = self.sigmoid(self.rate) * (torch.log(2.72 + (1 - self.sigmoid(dot_product))) * b_time)

        # e: [B, T]
        e = self.relu(self.sigmoid(dot_product) / denom)

        # Attention weights
        a = self.softmax(e)  # [B, T]

        # Weighted sum in the original hidden space
        v = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # [B, hidden_dim]
        return v, a


class FinalAttentionQKV(nn.Module):
    """
    Final attention over feature embeddings with optional mask-based biasing.
    """
    def __init__(self, hidden_dim, attention_hidden_dim, attention_type='mul', dropout=0.5, use_mask_bias=True):
        super().__init__()
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, attention_hidden_dim)
        self.W_k = nn.Linear(hidden_dim, attention_hidden_dim)
        self.W_v = nn.Linear(hidden_dim, attention_hidden_dim)
        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1))
        self.b_out = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.mask_scale = nn.Parameter(torch.tensor(1.0)) if use_mask_bias else None
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, feature_mask=None):
        """
        x: [B, N, hidden_dim] feature embeddings (including demographics if appended)
        feature_mask: optional [B, N] tensor with observation coverage per feature.
        Returns: pooled context [B, hidden_dim] and weights [B, F+1]
        """
        B, N, _ = x.size()

        q = self.W_q(x[:, -1, :])         # [B, H]
        k = self.W_k(x)                    # [B, N, H]
        v = self.W_v(x)                    # [B, N, H]

        if self.attention_type == 'mul':
            e = torch.matmul(k, q.unsqueeze(-1)).squeeze(-1)  # [B, N]
        elif self.attention_type == 'add':
            h = self.tanh(q.unsqueeze(1) + k + self.b_in)     # [B, N, H]
            e = self.W_out(h).squeeze(-1)                     # [B, N]
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

        if feature_mask is not None and self.mask_scale is not None:
            if feature_mask.shape != (B, N):
                raise ValueError(f"feature_mask must be shape [B, {N}] but got {feature_mask.shape}")
            e = e + self.mask_scale * feature_mask

        alpha = self.softmax(e)  # [B, N]
        alpha = self.dropout(alpha)
        pooled = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)  # [B, H]
        return pooled, alpha


class MultiHeadedAttention(nn.Module):
    """
    Multi-head self-attention with DeCov regularization.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        N = query.size(1)

        q = self.q_proj(query).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, v)  # [B, H, N, d_k]
        x = x.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.d_k)
        x = self.o_proj(x)  # [B, N, d_model]

        # DeCov regularization across features
        decov_loss = x.new_tensor(0.0)
        contexts = x.transpose(0, 1).transpose(1, 2)  # [N, d_model, B]
        if B > 1:
            for i in range(N):
                feats = contexts[i, :, :]  # [d_model, B]
                mean = feats.mean(dim=1, keepdim=True)
                centered = feats - mean
                cov = (1.0 / (B - 1)) * torch.matmul(centered, centered.t())
                decov_loss = decov_loss + 0.5 * (
                    torch.norm(cov, p='fro') ** 2 - torch.norm(torch.diag(cov)) ** 2
                )

        return x, decov_loss


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Pre-norm residual connection.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        out, aux = sublayer(self.norm(x))
        return x + self.dropout(out), aux


class ConCare(nn.Module):
    """
    Multi-channel GRU plus per-feature attention and feature pooling.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        d_model,
        MHD_num_head,
        d_ff,
        output_dim,
        keep_prob,
        demographic_dim,
        mask_dim: int = 0,
        base_value_dim: int | None = None,
        config_path: str | None = None,
        enable_missing_aware: bool = False,
        enable_mask_bias: bool = True,
        use_missing_temporal_attention: bool = True,
    ):
        super().__init__()

        self.use_missing_aware = enable_missing_aware
        self.use_missing_temporal_attention = use_missing_temporal_attention
        self.total_input_dim = input_dim
        if self.use_missing_aware:
            self.mask_dim = mask_dim
            self.value_dim = input_dim - mask_dim
            if self.value_dim <= 0:
                raise ValueError(f"input_dim ({input_dim}) must be larger than mask_dim ({mask_dim}).")
        else:
            self.mask_dim = 0
            self.value_dim = input_dim

        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.demographic_dim = demographic_dim
        self.base_value_dim = base_value_dim or self.value_dim

        if self.use_missing_aware:
            cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
            channel_names, mapping = _load_value_channel_mapping(cfg_path)
            fallback_channels = self.mask_dim if self.mask_dim > 0 else max(len(mapping), 1) or 1
            if not mapping:
                mapping = list(range(max(fallback_channels, 1)))
            if len(mapping) != self.value_dim:
                repeats = math.ceil(self.value_dim / len(mapping))
                mapping = (mapping * repeats)[:self.value_dim]
                warnings.warn(
                    f"Channel mapping length mismatch (expected {self.value_dim}, loaded {len(mapping)}). "
                    "Mapping has been truncated/repeated to align with value dimensions."
                )
            self.value_to_channel = mapping
            self.channel_count = len(channel_names) if channel_names else max(fallback_channels, 1)
        else:
            self.value_to_channel = []
            self.channel_count = 0

        # Per-feature GRUs
        self.GRUs = nn.ModuleList([nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.value_dim)])

        self.LastStepAttentions = nn.ModuleList([
            SingleAttentionPerFeatureNew(self.hidden_dim, attention_hidden_dim=8)
            for _ in range(self.value_dim)
        ])
        if self.use_missing_aware and self.use_missing_temporal_attention:
            self.TemporalAttentions = nn.ModuleList([
                MissingAwareTemporalAttention(self.hidden_dim, attention_hidden_dim=16)
                for _ in range(self.value_dim)
            ])
        else:
            self.TemporalAttentions = None

        # Demographic projections
        self.demo_proj_main = nn.Linear(self.demographic_dim, self.hidden_dim)
        self.demo_proj = nn.Linear(self.demographic_dim, self.hidden_dim)  # kept for parity with authors

        # Multi-head attention over feature embeddings
        self.MultiHeadedAttention = MultiHeadedAttention(self.d_model, self.MHD_num_head, dropout=0.1)

        # One SublayerConnection used twice
        self.SublayerConnection = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)

        # Positionwise feed-forward
        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)

        # Final attention pooling over features
        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type='mul',
            dropout=1 - self.keep_prob,
            use_mask_bias=self.use_missing_aware and enable_mask_bias,
        )

        # Output head
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, demo_input, return_context=False, context_only=False):
        """
        x: [B, T, F_total] where F_total=value_dim+mask_dim
        demo_input: [B, D]
        Returns: prediction [B, 1] and DeCov loss (or latent contexts if return_context=True)
        """
        B, T, F_total = x.size()
        device = x.device
        if F_total != self.total_input_dim:
            raise ValueError(f"Expected {self.total_input_dim} features, got {F_total}")
        assert self.d_model % self.MHD_num_head == 0

        # Demographics as an extra feature
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)  # [B, 1, H]

        value_feats = x[:, :, :self.value_dim]
        mask_feats = x[:, :, self.value_dim:] if self.mask_dim > 0 else None
        if mask_feats is not None and mask_feats.shape[-1] != self.mask_dim:
            raise ValueError(f"Mask tensor width mismatch: expected {self.mask_dim}, got {mask_feats.shape[-1]}")

        h0 = torch.zeros(1, B, self.hidden_dim, device=device)

        if self.use_missing_aware:
            feats = []
            feature_masks = []
            ones_mask = torch.ones(B, T, device=device)
            for i in range(self.value_dim):
                xi = value_feats[:, :, i].unsqueeze(-1)  # [B, T, 1]
                if mask_feats is None or self.mask_dim == 0:
                    mi = ones_mask
                else:
                    ch_idx = self.value_to_channel[i]
                    ch_idx = min(max(ch_idx, 0), self.mask_dim - 1)
                    mi = mask_feats[:, :, ch_idx]
                gru_out, _ = self.GRUs[i](xi, h0)
                if self.TemporalAttentions is not None:
                    vi, _ = self.TemporalAttentions[i](gru_out, mi)
                else:
                    vi, _ = self.LastStepAttentions[i](gru_out)
                feats.append(vi.unsqueeze(1))
                feature_masks.append(mi.mean(dim=1, keepdim=True))

            feats = torch.cat(feats, dim=1)
            feature_masks = torch.cat(feature_masks, dim=1)
            feats = torch.cat([feats, demo_main], dim=1)
            demo_mask = torch.ones(B, 1, device=device)
            feature_masks = torch.cat([feature_masks, demo_mask], dim=1)
            feats = self.dropout(feats)

            contexts, decov = self.SublayerConnection(feats, lambda z: self.MultiHeadedAttention(z, z, z, None))
            contexts = self.SublayerConnection(contexts, lambda z: self.PositionwiseFeedForward(z))[0]
            if return_context:
                return contexts, feature_masks, decov
            pooled, _ = self.FinalAttentionQKV(contexts, feature_masks)
        else:
            feats = []
            for i in range(self.value_dim):
                xi = value_feats[:, :, i].unsqueeze(-1)
                gru_out, _ = self.GRUs[i](xi, h0)
                vi, _ = self.LastStepAttentions[i](gru_out)
                feats.append(vi.unsqueeze(1))
            feats = torch.cat(feats, dim=1)
            feats = torch.cat([feats, demo_main], dim=1)
            feats = self.dropout(feats)
            contexts, decov = self.SublayerConnection(feats, lambda z: self.MultiHeadedAttention(z, z, z, None))
            contexts = self.SublayerConnection(contexts, lambda z: self.PositionwiseFeedForward(z))[0]
            if return_context:
                feature_masks = None
                return contexts, feature_masks, decov
            pooled, _ = self.FinalAttentionQKV(contexts)

        if context_only:
            return pooled, decov

        if context_only:
            return pooled, decov

        # Output
        out = self.output1(self.relu(self.output0(pooled)))
        out = self.sigmoid(out)  # [B, 1]
        return out, decov
