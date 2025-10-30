import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.u = nn.Linear(d_model, 1, bias=False)

    def forward(self, H):
        # H: [B, L, H]
        alpha = torch.softmax(self.u(torch.tanh(self.W(H))), dim=1).squeeze(-1)  # [B, L]
        f = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B, H]
        return f


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, L, D = Q.size()
        q = self.q_proj(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(K).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(V).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(attn, v)  # [B, H, L, d_k]
        out = head_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ConCare(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob, demographic_dim):
        super().__init__()

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.self_attention = MultiHeadedAttention(hidden_dim, MHD_num_head)
        self.feed_forward = FeedForward(hidden_dim, d_ff, dropout=1 - keep_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.single_attention = SingleAttention(hidden_dim)

        self.demographic_fc = nn.Linear(demographic_dim, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(1 - keep_prob)

    def forward(self, X, D):
        # X: [B, T, F] , D: [B, demo_dim]
        H = self.embed(X)
        attn_out = self.self_attention(H, H, H)
        H = self.layer_norm1(H + attn_out)
        ff_out = self.feed_forward(H)
        H = self.layer_norm2(H + ff_out)

        # attention pooling
        f = self.single_attention(H)

        # decorrelation loss (DeCov)
        U = H.view(-1, H.size(-1))
        C = (U.t() @ U) / max(U.shape[0], 1)
        I = torch.eye(C.shape[0], device=C.device)
        decov_loss = torch.sum((C * (1 - I)) ** 2)

        d_feat = torch.relu(self.demographic_fc(D))
        h = torch.cat([f, d_feat], dim=1)
        h = self.dropout(h)

        logits = self.fc_final(h)  # shape [B, 1]
        return logits, decov_loss
