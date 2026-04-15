import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)

        # Round up to multiple of 256
        hidden_features = (int(hidden_features * 2 / 3) + 255) // 256 * 256

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerEncoderLayerModern(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.feed_forward = SwiGLU(in_features=d_model, hidden_features=dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN architecture
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout2(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=11, hidden_dim=512, num_layers=4, nhead=8, max_size=30):
        super().__init__()
        # ARC digits 0-9 plus pad token (10)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=10)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_size * max_size, hidden_dim))

        self.layers = nn.ModuleList([
            TransformerEncoderLayerModern(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask):
        """
        x: (batch, max_size, max_size) containing values 0-9 or 10 (pad)
        mask: (batch, max_size, max_size) boolean mask where True is valid
        """
        b_sz, h, w = x.shape
        x_flat = x.view(b_sz, -1) # (batch, max_size * max_size)
        mask_flat = mask.view(b_sz, -1) # (batch, max_size * max_size)

        # In nn.TransformerEncoder, src_key_padding_mask requires True for *padded* elements
        # Our mask has True for *valid* elements, so we invert it
        padding_mask = ~mask_flat

        # Add CLS token mask (not padded)
        cls_mask = torch.zeros((b_sz, 1), dtype=torch.bool, device=x.device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        emb = self.embedding(x_flat) # (batch, seq_len, hidden)
        emb = emb + self.pos_embedding[:, :emb.size(1), :]

        cls_tokens = self.cls_token.expand(b_sz, -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)

        for layer in self.layers:
            emb = layer(emb, src_key_padding_mask=padding_mask)

        emb = self.norm(emb)

        # Return CLS token
        return emb[:, 0, :]

class ActionInference(nn.Module):
    def __init__(self, hidden_dim=512, num_actions=4, action_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        # We concatenate z_in and z_out, so input dim is hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            SwiGLU(hidden_dim, hidden_dim),
        )
        # Non-linear heads for each action's mu and logvar
        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_actions)
        ])
        self.logvar_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_actions)
        ])

    def forward(self, z_in, z_out):
        """
        z_in: (batch, hidden_dim)
        z_out: (batch, hidden_dim)
        """
        z_pair = torch.cat([z_in, z_out], dim=-1)
        features = self.fc(z_pair)
        mus = []
        logvars = []
        for i in range(self.num_actions):
            mus.append(self.mu_heads[i](features))
            logvars.append(self.logvar_heads[i](features))
        mus = torch.stack(mus, dim=1)       # (batch, num_actions, action_dim)
        logvars = torch.stack(logvars, dim=1) # (batch, num_actions, action_dim)
        return mus, logvars
        
    def sample(self, mus, logvars):
        stds = torch.exp(0.5 * logvars)
        eps = torch.randn_like(stds)
        return mus + eps * stds

class LatentPredictor(nn.Module):
    def __init__(self, hidden_dim=512, action_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            SwiGLU(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, z, a):
        """
        z: (batch, hidden_dim)
        a: (batch, action_dim)
        """
        x = torch.cat([z, a], dim=-1)
        # Residual connection
        return z + self.net(x)

class JEPARC(nn.Module):
    def __init__(self, vocab_size=11, hidden_dim=512, max_size=30, num_actions=4, action_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = TransformerEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim, max_size=max_size)
        self.action_inference = ActionInference(hidden_dim=hidden_dim, num_actions=num_actions, action_dim=action_dim)
        self.predictor = LatentPredictor(hidden_dim=hidden_dim, action_dim=action_dim)

    def forward(self, input_grid, input_mask, output_grid, output_mask):
        """
        input_grid, output_grid: (batch, num_pairs, max_size, max_size)
        """
        b_sz, n_pairs, h, w = input_grid.shape

        # Flatten batch and pairs
        input_grid_flat = input_grid.view(-1, h, w)
        input_mask_flat = input_mask.view(-1, h, w)
        output_grid_flat = output_grid.view(-1, h, w)
        output_mask_flat = output_mask.view(-1, h, w)

        # Encode
        z_in = self.encoder(input_grid_flat, input_mask_flat)
        z_out = self.encoder(output_grid_flat, output_mask_flat)

        # Action inference
        mus, logvars = self.action_inference(z_in, z_out)
        actions = self.action_inference.sample(mus, logvars) # (batch * num_pairs, num_actions, action_dim)

        # Rollout
        z_pred = z_in
        for i in range(self.num_actions):
            z_pred = self.predictor(z_pred, actions[:, i, :])

        # Reshape back to (batch, num_pairs, ...)
        z_in = z_in.view(b_sz, n_pairs, -1)
        z_out = z_out.view(b_sz, n_pairs, -1)
        z_pred = z_pred.view(b_sz, n_pairs, -1)
        mus = mus.view(b_sz, n_pairs, self.num_actions, -1)
        logvars = logvars.view(b_sz, n_pairs, self.num_actions, -1)
        actions = actions.view(b_sz, n_pairs, self.num_actions, -1)

        return z_in, z_out, z_pred, mus, logvars, actions
