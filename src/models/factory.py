from __future__ import annotations

from torch import nn

from src.models.informer import InformerForecaster
from src.models.lstm import LSTMForecaster
from src.models.tiny_time_mixer import TinyTimeMixerForecaster
from src.models.transformer import TransformerForecaster


def build_model(model_name: str, input_size: int, lookback_window: int, hidden_size: int) -> nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(input_size=input_size, hidden_size=hidden_size)
    if model_name == "transformer":
        return TransformerForecaster(input_size=input_size, hidden_size=hidden_size)
    if model_name == "tiny_time_mixer":
        return TinyTimeMixerForecaster(input_size=input_size, lookback_window=lookback_window, hidden_size=hidden_size)
    if model_name == "informer":
        return InformerForecaster(input_size=input_size, hidden_size=hidden_size)
    raise ValueError(f"Unsupported model: {model_name}")
