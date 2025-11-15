import torch
import torch.nn as nn

from .ConCare_Model_v3 import (
    SingleAttentionPerFeatureNew,
    MultiHeadedAttention,
    SublayerConnection,
    PositionwiseFeedForward,
    FinalAttentionQKV,
)


class ConCareDEMinus(nn.Module):
    """
    ConCareDE− ablation (paper Sec. 4): removes demographic embeddings while
    keeping multi-channel feature encoders and DeCov regularization intact.
    """

    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff,
                 output_dim, keep_prob, demographic_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # Per-feature GRUs plus time-aware attention (same as full ConCare)
        self.GRUs = nn.ModuleList([nn.GRU(1, hidden_dim, batch_first=True) for _ in range(input_dim)])
        self.LastStepAttentions = nn.ModuleList([
            SingleAttentionPerFeatureNew(hidden_dim, attention_hidden_dim=8)
            for _ in range(input_dim)
        ])

        # Multi-head attention across features (still uses DeCov)
        self.MultiHeadedAttention = MultiHeadedAttention(d_model, MHD_num_head, dropout=0.1)
        self.SublayerConnection = SublayerConnection(d_model, dropout=1 - keep_prob)
        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
        self.FinalAttentionQKV = FinalAttentionQKV(hidden_dim, hidden_dim, attention_type="mul",
                                                   dropout=1 - keep_prob)

        self.output0 = nn.Linear(hidden_dim, hidden_dim)
        self.output1 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=1 - keep_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, demo_input):  # noqa: ARG002 - demographics intentionally unused
        """
        Args:
            x: [B, T, F] dynamic feature tensor.
            demo_input: [B, D] demographic vectors (ignored per ablation design).
        Returns:
            probs: [B, 1] sigmoid outputs.
            decov: decorrelation loss from feature self-attention.
        """
        batch_size, time_steps, feature_dim = x.size()
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {feature_dim}")

        device = x.device
        feature_vectors = []
        for feat_idx in range(self.input_dim):
            signal = x[:, :, feat_idx].unsqueeze(-1)
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            gru_out, _ = self.GRUs[feat_idx](signal, h0)
            vi, _ = self.LastStepAttentions[feat_idx](gru_out)
            feature_vectors.append(vi.unsqueeze(1))

        feats = torch.cat(feature_vectors, dim=1)
        feats = self.dropout(feats)

        contexts, decov = self.SublayerConnection(feats, lambda _z: self.MultiHeadedAttention(feats, feats, feats, None))
        contexts = self.SublayerConnection(contexts, lambda _z: self.PositionwiseFeedForward(contexts))[0]
        pooled, _ = self.FinalAttentionQKV(contexts)

        logits = self.output1(self.relu(self.output0(pooled)))
        probs = self.sigmoid(logits)
        return probs, decov
