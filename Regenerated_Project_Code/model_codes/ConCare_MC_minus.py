import torch
import torch.nn as nn


class VisitSelfAttention(nn.Module):
    """
    Single-head additive attention over visit-level hidden states.
    Used to summarize the sequence into a health-status embedding.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=5 ** 0.5)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: [B, T, H] sequence of visit representations
        Returns:
            context: [B, H] pooled embedding
            alpha:   [B, T] attention weights across visits
        """
        scores = self.score(torch.tanh(self.proj(H))).squeeze(-1)  # [B, T]
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return context, alpha


class ConCareMCMinus(nn.Module):
    """
    ConCareMC- ablation (paper Sec. 4): visit-level health embedding only, no DeCov.

    Differences from the full ConCare model:
      * Visits are encoded with a single GRU instead of 76 per-feature channels.
      * Static demographics are fused after sequence pooling (no feature-level self-attention).
      * The decorrelation loss is removed (always returns 0).
    """

    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob, demographic_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.demographic_dim = demographic_dim

        # Visit-level encoder (shared across all features)
        self.visit_proj = nn.Linear(input_dim, hidden_dim)
        self.visit_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.visit_attention = VisitSelfAttention(hidden_dim)

        # Demographic pathway
        self.demo_proj = nn.Linear(demographic_dim, hidden_dim)

        # Fusion + prediction head
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=1 - keep_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, visits: torch.Tensor, demo_input: torch.Tensor):
        """
        Args:
            visits: [B, T, F] tensor of visit-level measurements
            demo_input: [B, D] demographic / static baseline features
        Returns:
            probs: [B, 1] sigmoid outputs
            decov: scalar tensor (always 0, since this ablation disables DeCov)
        """
        batch_size, _, feature_dim = visits.size()
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {feature_dim}")

        device = visits.device
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        visit_embed = self.tanh(self.visit_proj(visits))
        visit_hidden, _ = self.visit_gru(visit_embed, h0)
        visit_context, _ = self.visit_attention(visit_hidden)

        demo_context = self.relu(self.demo_proj(demo_input))

        fused = torch.cat([visit_context, demo_context], dim=1)
        fused = self.dropout(self.relu(self.fusion(fused)))
        logits = self.output(fused)
        probs = self.sigmoid(logits)

        decov = torch.zeros(1, device=device, dtype=probs.dtype)
        return probs, decov
