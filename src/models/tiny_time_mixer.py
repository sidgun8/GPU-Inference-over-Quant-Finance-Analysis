from __future__ import annotations

import torch
from torch import nn


class TinyTimeMixerForecaster(nn.Module):
    def __init__(self, input_size: int, lookback_window: int, hidden_size: int = 64, output_size: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * lookback_window, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
