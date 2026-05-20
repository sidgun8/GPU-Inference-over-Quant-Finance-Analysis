from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_or_load_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    x_train,
    y_train,
    device: torch.device,
    train_if_missing: bool,
    epochs: int,
) -> nn.Module:
    path = Path(checkpoint_path)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device).eval()
    if not train_if_missing:
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), path)
    return model.eval()
