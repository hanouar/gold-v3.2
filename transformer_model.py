from __future__ import annotations
import torch
from torch import nn

class TemporalTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dropout: float = 0.15, n_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        # x: [batch, seq, features]
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)
