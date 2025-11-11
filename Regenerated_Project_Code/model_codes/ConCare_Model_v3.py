import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        # registers empty buffers that will be filled on the first forward pass
        self.register_buffer("saved_attn_avg", torch.zeros(0))            # will become [H, N, N]
        self.register_buffer("saved_attn_count", torch.zeros((), dtype=torch.float32))

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
        
        # --- store a running average of attention across batches ---
        # CRITICAL: Save BEFORE dropout to preserve row-normalized attention
        with torch.no_grad():
            cur = p_attn.detach().mean(dim=0)  # [H, N, N] average over batch
            if self.saved_attn_avg.numel() == 0:
                self.saved_attn_avg = cur.clone()
            else:
                self.saved_attn_avg = (self.saved_attn_avg * self.saved_attn_count + cur) / (self.saved_attn_count + 1.0)
            self.saved_attn_count += 1.0
        # -----------------------------------------------------------

        # Apply dropout AFTER saving attention
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
    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob, demographic_dim):
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

        # Per-feature GRUs
        self.GRUs = nn.ModuleList([nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.input_dim)])

        # Per-feature attention using authors' 'new' time-aware formulation
        self.LastStepAttentions = nn.ModuleList([
            SingleAttentionPerFeatureNew(self.hidden_dim, attention_hidden_dim=8)
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
        for i in range(F):
            xi = x[:, :, i].unsqueeze(-1)  # [B, T, 1]
            h0 = torch.zeros(1, B, self.hidden_dim, device=device)
            gru_out, _ = self.GRUs[i](xi, h0)            # [B, T, H]
            vi, _ = self.LastStepAttentions[i](gru_out)  # [B, H]
            feats.append(vi.unsqueeze(1))                # [B, 1, H]

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
