# concare.py
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleAttention(nn.Module):
    """
    New time-aware single-head attention over a per-feature sequence.

    - Learnable projections Wt (for query from the last hidden state) and Wx (for keys over all steps)
    - dot = <Wt(h_T), Wx(h_t)>
    - tdecay = [1, 2, ..., T] so the most recent step has decay 1
    - denom = sigmoid(rate) * log(2.72 + (1 - sigmoid(dot))) * tdecay
    - e_t = ReLU(sigmoid(dot) / denom)
    - alpha = softmax(e)
    - output is the weighted sum over the original GRU hidden sequence H
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Wt = nn.Linear(hidden_dim, hidden_dim, bias=True)  # query from last state
        self.Wx = nn.Linear(hidden_dim, hidden_dim, bias=True)  # keys from each step
        self.rate = nn.Parameter(torch.tensor(0.0))  # learnable scalar, passed through sigmoid

        # init
        nn.init.kaiming_uniform_(self.Wt.weight, a=math.sqrt(5))
        nn.init.zeros_(self.Wt.bias)
        nn.init.kaiming_uniform_(self.Wx.weight, a=math.sqrt(5))
        nn.init.zeros_(self.Wx.bias)

    def forward(
        self,
        H: torch.Tensor,                 # [B, T, H]
        _: Optional[torch.Tensor] = None # delta_t not used since internal positional decay is required
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, Hdim = H.shape

        # Query is the last hidden state projected by Wt
        q = self.Wt(H[:, -1, :]).unsqueeze(1)           # [B, 1, H]

        # Keys are all time steps projected by Wx
        K = self.Wx(H)                                  # [B, T, H]

        # Dot score per step
        dot = (q * K).sum(dim=-1)                       # [B, T]

        # Positional decay so that the most recent step has decay 1
        # Build [T,] as [T, T-1, ..., 1] then reverse to [1, 2, ..., T] aligned with time dimension
        # Since H is in chronological order, we want last step to have 1
        tdecay = torch.arange(T, 0, -1, device=H.device, dtype=H.dtype)  # [T,] gives [T..1]
        tdecay = torch.flip(tdecay, dims=[0])                             # [1..T]
        tdecay = tdecay.view(1, T).repeat(B, 1)                           # [B, T]

        # Denominator
        rate_sigma = torch.sigmoid(self.rate)                             # scalar in (0,1)
        dot_sigma = torch.sigmoid(dot)                                    # [B, T]
        denom = rate_sigma * torch.log(torch.tensor(2.72, device=H.device, dtype=H.dtype) + (1.0 - dot_sigma)) * tdecay
        denom = torch.clamp(denom, min=1e-6)

        # Energy and attention
        e = F.relu(dot_sigma / denom)                                     # [B, T]
        alpha = F.softmax(e, dim=1)                                       # [B, T]

        # Weighted sum over the ORIGINAL H
        f = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)                   # [B, H]
        return f, alpha


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(F.relu(self.lin1(x))))


class SublayerConnection(nn.Module):
    """
    Pre-norm residual block: x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-7)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))


class MultiHeadedAttentionWithDeCov(nn.Module):
    """
    Standard multi head self attention over tokens.
    Returns output and a decorrelation loss term computed across head outputs.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.h, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        k = self.W_k(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)   # [B, H, L, L]
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        head_out = torch.matmul(attn, v)                                     # [B, H, L, d_k]

        out = head_out.transpose(1, 2).contiguous().view(B, L, D)            # [B, L, D]
        out = self.W_o(out)                                                  # [B, L, D]

        # DeCov loss on concatenated heads per position
        U = head_out.transpose(1, 2).contiguous().view(B * L, self.h * self.d_k) # [B*L, D]
        U = U - U.mean(dim=0, keepdim=True)
        C = (U.t() @ U) / max(U.shape[0], 1)
        decov = 0.5 * (C.pow(2).sum() - C.diag().pow(2).sum())

        return out, decov


class FinalAttentionQKV(nn.Module):
    """
    Attention readout over tokens to a single patient vector.
    Multiplicative scoring with a linear projection to scalar.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.W_q = nn.Linear(d_model, hidden_dim)
        self.W_k = nn.Linear(d_model, hidden_dim)
        self.W_v = nn.Linear(d_model, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, L, D]
        Q = self.W_q(x)      # [B, L, H]
        K = self.W_k(x)      # [B, L, H]
        V = self.W_v(x)      # [B, L, H]

        scores = self.W_out(Q * K).squeeze(-1)  # [B, L]
        alpha = F.softmax(scores, dim=1)        # [B, L]
        s = torch.bmm(alpha.unsqueeze(1), V).squeeze(1)  # [B, H]
        return s, alpha


class ConCare(nn.Module):
    """
    ConCare model in PyTorch 2.x

    Args
        input_dim: number of dynamic features N
        hidden_dim: token size and d_model
        d_model: equal to hidden_dim
        MHD_num_head: number of attention heads
        d_ff: inner size of FFN
        output_dim: 1 for binary
        keep_prob: keep probability used to set dropout
        demographic_dim: fixed default 12 and enforced in forward
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_model: int,
        MHD_num_head: int,
        d_ff: int,
        output_dim: int,
        keep_prob: float = 0.5,
        demographic_dim: int = 12,
    ):
        super().__init__()
        assert d_model == hidden_dim
        self.N = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = 1.0 - keep_prob
        self.demographic_dim = demographic_dim  # default 12

        # one GRU per feature
        self.grus = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)
            for _ in range(input_dim)
        ])
        # one new time aware attention per feature
        self.time_atts = nn.ModuleList([
            SingleAttention(hidden_dim)
            for _ in range(input_dim)
        ])

        # demographic projection used as an extra token
        self.demo_proj_main = nn.Linear(demographic_dim, hidden_dim)
        self.token_drop = nn.Dropout(self.dropout_p)

        # encoder over tokens
        self.mh_attn = MultiHeadedAttentionWithDeCov(d_model, MHD_num_head, dropout=self.dropout_p)
        self.sublayer1 = SublayerConnection(d_model, dropout=self.dropout_p)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=self.dropout_p)
        self.sublayer2 = SublayerConnection(d_model, dropout=self.dropout_p)

        # final readout
        self.final_attn = FinalAttentionQKV(d_model, hidden_dim, dropout=self.dropout_p)

        # output head
        self.out0 = nn.Linear(hidden_dim, hidden_dim)
        self.out1 = nn.Linear(hidden_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # GRU init
        for gru in self.grus:
            for name, param in gru.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif "bias" in name:
                    nn.init.zeros_(param)
        nn.init.kaiming_uniform_(self.demo_proj_main.weight, a=math.sqrt(5))
        nn.init.zeros_(self.demo_proj_main.bias)
        nn.init.kaiming_uniform_(self.out0.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out0.bias)
        nn.init.kaiming_uniform_(self.out1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out1.bias)

    def forward(
        self,
        X: torch.Tensor,                         # [B, T, N]
        D: torch.Tensor,                         # [B, 12] enforced
        delta_t: Optional[torch.Tensor] = None,  # accepted for API symmetry but ignored by SingleAttention
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
            y_prob: [B, 1]
            decov_loss: scalar decorrelation loss to add to BCE
        """
        B, T, N = X.shape
        assert N == self.N, f"Expected {self.N} features but got {N}"
        assert D.shape[1] == self.demographic_dim == 12, "Demographic_dim must be 12"

        tokens: List[torch.Tensor] = []
        # loop over features
        for i in range(N):
            xi = X[:, :, i:i+1]                           # [B, T, 1]
            Hi, _ = self.grus[i](xi)                      # [B, T, H]
            fi, _ = self.time_atts[i](Hi, None)           # [B, H]
            tokens.append(fi)

        F_dyn = torch.stack(tokens, dim=1)                # [B, N, H]
        f_demo = torch.tanh(self.demo_proj_main(D)).unsqueeze(1)  # [B, 1, H]
        F_in = torch.cat([F_dyn, f_demo], dim=1)          # [B, N+1, H]
        F_in = self.token_drop(F_in)

        # encoder block with decorrelation
        y1, decov_loss = self.mh_attn(F_in)
        Y = self.sublayer1(F_in, lambda z: self.mh_attn(z)[0])
        Z = self.sublayer2(Y, self.ffn)

        # final attention readout
        s, alpha = self.final_attn(Z)                     # [B, H]
        h = F.relu(self.out0(s))
        logits = self.out1(h)                             # [B, 1]
        y_prob = torch.sigmoid(logits)
        return y_prob, decov_loss