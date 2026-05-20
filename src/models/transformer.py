from __future__ import annotations

import torch
from torch import nn


class TransformerForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_heads: int = 4, num_layers: int = 2, output_size: int = 1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])
