import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleAttentionPerFeature(nn.Module):
    """
    Time-aware attention for individual features (authors' LastStepAttentions).
    Simplified version of authors' 'new' attention type with time decay.
    """
    def __init__(self, hidden_dim, attention_hidden_dim=8, time_aware=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.time_aware = time_aware
        
        # Query from last timestep, Key from all timesteps
        self.Wt = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        self.Wx = nn.Parameter(torch.randn(hidden_dim, attention_hidden_dim))
        
        # Time-aware parameter
        if self.time_aware:
            self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
            nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
        
        self.bh = nn.Parameter(torch.zeros(attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, H):
        """
        H: [B, T, hidden_dim] - GRU outputs for one feature
        Returns: [B, hidden_dim] - attended representation
        """
        batch_size, time_step, _ = H.size()
        device = H.device
        
        # Query from last timestep
        q = torch.matmul(H[:, -1, :], self.Wt)  # [B, attn_hidden]
        q = q.unsqueeze(1)  # [B, 1, attn_hidden]
        
        # Keys from all timesteps
        k = torch.matmul(H, self.Wx)  # [B, T, attn_hidden]
        
        # Time decay (simplified - counts backwards from most recent)
        if self.time_aware:
            # Create time decay: [47, 46, ..., 1, 0] for 48 timesteps
            time_decays = torch.arange(time_step - 1, -1, -1, dtype=torch.float32, device=device)
            time_decays = time_decays.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # [B, T, 1], add 1 to avoid zero
            time_hidden = torch.matmul(time_decays, self.Wtime_aware)  # [B, T, attn_hidden]
            h = q + k + self.bh + time_hidden
        else:
            h = q + k + self.bh
        
        h = self.tanh(h)  # [B, T, attn_hidden]
        e = torch.matmul(h, self.Wa) + self.ba  # [B, T, 1]
        e = e.squeeze(-1)  # [B, T]
        
        # Attention weights
        alpha = self.softmax(e)  # [B, T]
        
        # Weighted sum
        v = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B, hidden_dim]
        
        return v, alpha


class FinalAttentionQKV(nn.Module):
    """
    Final attention over feature embeddings (authors' FinalAttentionQKV).
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
    
    def forward(self, input):
        """
        input: [B, F+1, hidden_dim] where F is num features + 1 for demographics
        Returns: [B, hidden_dim] - final representation
        """
        batch_size, num_features, _ = input.size()
        
        # Project to Q, K, V
        input_q = self.W_q(input[:, -1, :])  # [B, attn_hidden] - query from last (demo) feature
        input_k = self.W_k(input)  # [B, F+1, attn_hidden]
        input_v = self.W_v(input)  # [B, F+1, attn_hidden]
        
        if self.attention_type == 'mul':
            # Multiplicative attention
            q = input_q.unsqueeze(-1)  # [B, attn_hidden, 1]
            e = torch.matmul(input_k, q).squeeze(-1)  # [B, F+1]
        elif self.attention_type == 'add':
            # Additive attention
            q = input_q.unsqueeze(1)  # [B, 1, attn_hidden]
            h = q + input_k + self.b_in  # [B, F+1, attn_hidden]
            h = self.tanh(h)
            e = self.W_out(h).squeeze(-1)  # [B, F+1]
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")
        
        # Attention weights
        alpha = self.softmax(e)  # [B, F+1]
        if self.dropout is not None:
            alpha = self.dropout(alpha)
        
        # Weighted sum
        v = torch.bmm(alpha.unsqueeze(1), input_v).squeeze(1)  # [B, attn_hidden]
        
        return v, alpha


class MultiHeadedAttention(nn.Module):
    """
    Multi-head self-attention with DeCov regularization (authors' implementation).
    """
    def __init__(self, d_model, num_heads, dropout=0.5):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Separate projections for Q, K, V (unlike authors who use clones)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        num_features = query.size(1)  # F+1
        
        # Linear projections and reshape to [B, H, F+1, d_k]
        q = self.q_proj(query).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, num_features, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, F+1, F+1]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = F.softmax(scores, dim=-1)  # [B, H, F+1, F+1]
        p_attn = self.dropout(p_attn)
        
        # Apply attention to values
        x = torch.matmul(p_attn, v)  # [B, H, F+1, d_k]
        
        # Concatenate heads and project
        x = x.transpose(1, 2).contiguous().view(batch_size, num_features, self.num_heads * self.d_k)
        x = self.o_proj(x)  # [B, F+1, d_model]
        
        # DeCov regularization (authors' implementation)
        # Compute covariance matrix for each feature dimension
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # [F+1, d_model, B]
        DeCov_loss = 0.0
        for i in range(num_features):
            feature_activations = DeCov_contexts[i, :, :]  # [d_model, B]
            # Compute covariance
            mean = feature_activations.mean(dim=1, keepdim=True)
            centered = feature_activations - mean
            cov = (1.0 / (batch_size - 1)) * torch.matmul(centered, centered.t())
            # Penalize off-diagonal elements
            DeCov_loss += 0.5 * (torch.norm(cov, p='fro') ** 2 - torch.norm(torch.diag(cov)) ** 2)
        
        return x, DeCov_loss


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network (authors' implementation)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None


class LayerNorm(nn.Module):
    """Layer normalization (authors' custom implementation)."""
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
    Residual connection followed by layer norm (authors' implementation).
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class ConCare(nn.Module):
    """
    ConCare model with multi-channel GRU architecture (matching authors' implementation).
    
    Key changes from simplified version:
    - Uses separate GRU for each input feature (76 GRUs)
    - Uses per-feature attention mechanisms
    - Processes demographics as an additional "feature"
    - Implements full DeCov regularization
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
        
        # CRITICAL: One GRU per input feature (multi-channel architecture)
        self.GRUs = nn.ModuleList([
            nn.GRU(1, self.hidden_dim, batch_first=True) 
            for _ in range(self.input_dim)
        ])
        
        # CRITICAL: One attention module per feature
        self.LastStepAttentions = nn.ModuleList([
            SingleAttentionPerFeature(self.hidden_dim, attention_hidden_dim=8, time_aware=True)
            for _ in range(self.input_dim)
        ])
        
        # Final attention over all feature embeddings
        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim, 
            self.hidden_dim, 
            attention_type='mul',
            dropout=1 - self.keep_prob
        )
        
        # Multi-head self-attention
        self.MultiHeadedAttention = MultiHeadedAttention(
            self.d_model, 
            self.MHD_num_head, 
            dropout=1 - self.keep_prob
        )
        
        # Residual connections
        self.SublayerConnection1 = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)
        self.SublayerConnection2 = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)
        
        # Feed-forward network
        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.d_model, 
            self.d_ff, 
            dropout=0.1
        )
        
        # Demographic projection
        self.demo_proj_main = nn.Linear(self.demographic_dim, self.hidden_dim)
        
        # Output layers
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input, demo_input):
        """
        Args:
            input: [batch_size, timestep, feature_dim] - temporal features
            demo_input: [batch_size, demographic_dim] - demographic features
        
        Returns:
            output: [batch_size, 1] - prediction (sigmoid applied)
            DeCov_loss: scalar - decorrelation loss
        """
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        device = input.device
        
        assert feature_dim == self.input_dim, f"Expected {self.input_dim} features, got {feature_dim}"
        assert self.d_model % self.MHD_num_head == 0
        
        # Process demographics
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # CRITICAL: Process each feature through its own GRU + Attention
        # This is the main architectural difference from the simplified version
        feature_embeddings = []
        
        for i in range(feature_dim):
            # Extract single feature as [B, T, 1]
            feature_i = input[:, :, i].unsqueeze(-1)
            
            # Initialize hidden state
            h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            
            # Pass through feature-specific GRU
            gru_output, _ = self.GRUs[i](feature_i, h_0)  # [B, T, hidden_dim]
            
            # Apply feature-specific attention
            attended_feature, _ = self.LastStepAttentions[i](gru_output)  # [B, hidden_dim]
            
            # Add to list
            feature_embeddings.append(attended_feature.unsqueeze(1))  # [B, 1, hidden_dim]
        
        # Stack all feature embeddings [B, F, hidden_dim]
        Attention_embeded_input = torch.cat(feature_embeddings, dim=1)
        
        # Add demographics as an additional "feature"
        Attention_embeded_input = torch.cat([Attention_embeded_input, demo_main], dim=1)  # [B, F+1, hidden_dim]
        
        # Apply dropout
        posi_input = self.dropout(Attention_embeded_input)  # [B, F+1, hidden_dim]
        
        # Multi-head self-attention with residual connection
        contexts, DeCov_loss = self.SublayerConnection1(
            posi_input, 
            lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None)
        )
        
        # Feed-forward with residual connection
        contexts = self.SublayerConnection2(contexts, lambda x: self.PositionwiseFeedForward(contexts))[0]
        
        # Final attention pooling over features
        weighted_contexts, _ = self.FinalAttentionQKV(contexts)
        
        # Output projection
        output = self.output1(self.relu(self.output0(weighted_contexts)))  # [B, 1]
        output = self.sigmoid(output)
        
        return output, DeCov_loss
