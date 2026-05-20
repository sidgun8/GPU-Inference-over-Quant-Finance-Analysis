from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_supervised_arrays(
    csv_paths: dict[str, Path],
    features: list[str],
    target: str,
    lookback_window: int,
    forecast_horizon: int,
    train_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for path in csv_paths.values():
        frame = pd.read_csv(path)
        selected_columns = list(dict.fromkeys([*features, target]))
        missing = [column for column in selected_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        frame = frame[selected_columns].dropna()
        feature_values = frame[features].to_numpy(dtype=np.float32)
        target_values = frame[[target]].to_numpy(dtype=np.float32)
        for index in range(lookback_window, len(frame) - forecast_horizon + 1):
            xs.append(feature_values[index - lookback_window:index])
            ys.append(target_values[index + forecast_horizon - 1])
    if not xs:
        raise RuntimeError("No supervised samples were created. Check lookback window, horizon, and data length.")

    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    split_index = int(len(x) * train_split)
    x_train_raw, x_test_raw = x[:split_index], x[split_index:]
    y_train_raw, y_test_raw = y[:split_index], y[split_index:]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train_flat = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    feature_scaler.fit(x_train_flat)
    target_scaler.fit(y_train_raw)

    x_train = feature_scaler.transform(x_train_flat).reshape(x_train_raw.shape).astype(np.float32)
    x_test = feature_scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape).astype(np.float32)
    y_train = target_scaler.transform(y_train_raw).astype(np.float32)
    y_test = target_scaler.transform(y_test_raw).astype(np.float32)
    return x_train, y_train, x_test, y_test, feature_scaler, target_scaler
