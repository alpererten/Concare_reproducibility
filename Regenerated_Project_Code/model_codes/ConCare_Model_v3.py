import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding used by Transformer encoders.
    """

    def __init__(self, hidden_dim, dropout=0.0, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))

        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, x):
        """
        x: [B, T, H]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class SingleAttentionPerFeatureNew(nn.Module):
    """
    Time-aware attention per feature using the authors' 'new' formulation.
    Matches their rate-controlled nonlinear time decay.
    """
    def __init__(self, hidden_dim, attention_hidden_dim=8, time_aware=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.time_aware = bool(time_aware)

        # Query from last timestep, Key from all timesteps
        self.Wt = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        self.Wx = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        # Learned rate that controls the time decay strength (when enabled)
        if self.time_aware:
            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
        else:
            self.register_parameter("rate", None)

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

        # Dot attention scores across time: [B, T]
        dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze(1)

        if self.time_aware:
            # Time indices from most recent back: [B, T]
            b_time = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1) + 1.0

            # Authors' nonlinear time-aware normalization: [B, T]
            denom = self.sigmoid(self.rate) * (torch.log(2.72 + (1 - self.sigmoid(dot_product))) * b_time)

            # e: [B, T]
            e = self.relu(self.sigmoid(dot_product) / denom)
        else:
            # Pure dot-product attention when time-aware decay is disabled
            e = dot_product

        # Attention weights
        a = self.softmax(e)  # [B, T]

        # Weighted sum in the original hidden space
        v = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # [B, hidden_dim]
        return v, a


class FinalAttentionQKV(nn.Module):
    """
    Final attention over feature embeddings.
    """
    def __init__(self, hidden_dim, attention_hidden_dim, attention_type='mul', dropout=0.5):
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
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: [B, F+1, hidden_dim] where F is number of features and +1 is demographics
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
        decov_loss = 0.0
        contexts = x.transpose(0, 1).transpose(1, 2)  # [N, d_model, B]
        for i in range(N):
            feats = contexts[i, :, :]  # [d_model, B]
            mean = feats.mean(dim=1, keepdim=True)
            centered = feats - mean
            cov = (1.0 / (B - 1)) * torch.matmul(centered, centered.t())
            decov_loss += 0.5 * (torch.norm(cov, p='fro') ** 2 - torch.norm(torch.diag(cov)) ** 2)

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
        time_aware_attention=True,
        sequence_encoder: str = "gru",
        positional_layers: int = 2,
        positional_heads: int = 4,
        positional_dropout: float = 0.1,
    ):
        super().__init__()

        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.demographic_dim = demographic_dim
        sequence_encoder = sequence_encoder.lower()
        if sequence_encoder not in {"gru", "positional"}:
            raise ValueError(f"sequence_encoder must be 'gru' or 'positional' (got {sequence_encoder})")
        self.sequence_encoder = sequence_encoder
        self.time_aware_attention = bool(time_aware_attention and self.sequence_encoder == "gru")

        # Per-feature encoders
        if self.sequence_encoder == "gru":
            self.GRUs = nn.ModuleList([nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.input_dim)])
            self.value_projs = nn.ModuleList()
            self.per_feature_transformer = None
            self.positional_encoding = None
        else:
            self.GRUs = nn.ModuleList()
            self.value_projs = nn.ModuleList([nn.Linear(1, self.hidden_dim) for _ in range(self.input_dim)])
            if self.hidden_dim % positional_heads != 0:
                raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by positional_heads ({positional_heads})")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=positional_heads,
                dim_feedforward=self.d_ff,
                dropout=positional_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.per_feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers=positional_layers)
            self.positional_encoding = PositionalEncoding(self.hidden_dim, dropout=positional_dropout)

        # Per-feature attention using authors' 'new' formulation (optionally time-aware)
        self.LastStepAttentions = nn.ModuleList([
            SingleAttentionPerFeatureNew(
                self.hidden_dim,
                attention_hidden_dim=8,
                time_aware=self.time_aware_attention,
            )
            for _ in range(self.input_dim)
        ])

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
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul', dropout=1 - self.keep_prob)

        # Output head
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, demo_input):
        """
        x: [B, T, F]
        demo_input: [B, D]
        Returns: prediction [B, 1] and DeCov loss
        """
        B, T, F = x.size()
        device = x.device
        assert F == self.input_dim, f"Expected {self.input_dim} features, got {F}"
        assert self.d_model % self.MHD_num_head == 0

        # Demographics as an extra feature
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)  # [B, 1, H]

        # Per-feature GRU + attention
        feats = []
        if self.sequence_encoder == "gru":
            for i in range(F):
                xi = x[:, :, i].unsqueeze(-1)  # [B, T, 1]
                h0 = torch.zeros(1, B, self.hidden_dim, device=device)
                gru_out, _ = self.GRUs[i](xi, h0)            # [B, T, H]
                vi, _ = self.LastStepAttentions[i](gru_out)  # [B, H]
                feats.append(vi.unsqueeze(1))                # [B, 1, H]
        else:
            for i in range(F):
                xi = x[:, :, i].unsqueeze(-1)               # [B, T, 1]
                proj = self.value_projs[i](xi)              # [B, T, H]
                proj = self.positional_encoding(proj)       # [B, T, H]
                enc = self.per_feature_transformer(proj)    # [B, T, H]
                vi, _ = self.LastStepAttentions[i](enc)     # [B, H]
                feats.append(vi.unsqueeze(1))               # [B, 1, H]

        feats = torch.cat(feats, dim=1)                  # [B, F, H]
        feats = torch.cat([feats, demo_main], dim=1)     # [B, F+1, H]
        feats = self.dropout(feats)

        # Self-attention over features with DeCov
        contexts, decov = self.SublayerConnection(feats, lambda z: self.MultiHeadedAttention(feats, feats, feats, None))

        # Feed-forward with the same SublayerConnection
        contexts = self.SublayerConnection(contexts, lambda z: self.PositionwiseFeedForward(contexts))[0]

        # Final attention pooling
        pooled, _ = self.FinalAttentionQKV(contexts)

        # Output
        out = self.output1(self.relu(self.output0(pooled)))
        out = self.sigmoid(out)  # [B, 1]
        return out, decov
