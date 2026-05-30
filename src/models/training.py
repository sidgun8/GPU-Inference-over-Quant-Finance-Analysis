from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    validation_split: float = 0.15,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
    save_training_history: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path)
    history_path = path.with_suffix(".history.json")
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        metadata: dict[str, Any] = {
            "checkpoint_path": str(path),
            "checkpoint_loaded": True,
            "training_history_path": str(history_path) if history_path.exists() else None,
        }
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                saved_history = json.load(handle)
            metadata.update({
                "trained_epochs": saved_history.get("trained_epochs"),
                "best_epoch": saved_history.get("best_epoch"),
                "best_val_loss": saved_history.get("best_val_loss"),
            })
        return model.to(device).eval(), metadata
    if not train_if_missing:
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    validation_count = int(len(x_train) * validation_split)
    if validation_count <= 0 or validation_count >= len(x_train):
        raise ValueError(f"validation_split={validation_split} produced invalid validation_count={validation_count} for {len(x_train)} samples.")
    train_count = len(x_train) - validation_count
    train_dataset = TensorDataset(torch.from_numpy(x_train[:train_count]), torch.from_numpy(y_train[:train_count]))
    validation_dataset = TensorDataset(torch.from_numpy(x_train[train_count:]), torch.from_numpy(y_train[train_count:]))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_x.shape[0])
            train_loss_total += float(loss.detach().cpu()) * batch_size
            train_samples += batch_size
        model.eval()
        validation_loss_total = 0.0
        validation_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = loss_fn(model(batch_x), batch_y)
                batch_size = int(batch_x.shape[0])
                validation_loss_total += float(loss.detach().cpu()) * batch_size
                validation_samples += batch_size
        train_loss = train_loss_total / train_samples
        validation_loss = validation_loss_total / validation_samples
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss})
        if validation_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), path)
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_loaded": False,
        "training_history_path": str(history_path) if save_training_history else None,
        "trained_epochs": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation_split": validation_split,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    if save_training_history:
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump({**metadata, "history": history}, handle, indent=2)
    return model.eval(), metadata
