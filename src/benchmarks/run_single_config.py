from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.fetch_yfinance import fetch_yfinance_data
from src.data.preprocessing import build_supervised_arrays
from src.inference.mlx_runner import run_mlx_inference_benchmark
from src.inference.onnx_runner import run_onnx_inference_benchmark
from src.inference.pytorch_runner import run_pytorch_inference_benchmark
from src.inference.tensorrt_runner import run_tensorrt_inference_benchmark
from src.models.factory import build_model
from src.models.training import train_or_load_model
from src.utils.cleanup import cleanup_cuda
from src.utils.config import load_config
from src.utils.device import get_hardware_metadata, require_cuda_device
from src.utils.validation import validate_config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_config(config: dict[str, Any], model_name: str, backend: str, precision: str, batch_size: int, repeat_index: int) -> dict[str, Any]:
    validate_config(config)
    set_seed(int(config["benchmark"].get("seed", 42)))
    device = require_cuda_device(config["hardware"].get("allow_cpu_fallback", False))
    cleanup_cuda(reset_memory_stats=config["benchmark"].get("reset_memory_stats_between_runs", True))

    dataset_config = config["dataset"]
    raw_paths = fetch_yfinance_data(
        tickers=dataset_config["tickers"],
        start_date=dataset_config["start_date"],
        end_date=dataset_config.get("end_date"),
        output_dir=Path("data") / "raw",
    )
    x_train, y_train, x_test, y_test, _, _ = build_supervised_arrays(
        csv_paths=raw_paths,
        features=dataset_config["features"],
        target=dataset_config["target"],
        lookback_window=int(dataset_config["lookback_window"]),
        forecast_horizon=int(dataset_config["forecast_horizon"]),
        train_split=float(dataset_config["train_split"]),
    )

    input_size = x_train.shape[-1]
    model = build_model(
        model_name=model_name,
        input_size=input_size,
        lookback_window=int(dataset_config["lookback_window"]),
        hidden_size=int(config["models"]["hidden_size"]),
    )
    checkpoint_path = (
        Path(config["models"]["checkpoint_dir"])
        / f"{model_name}_lb{int(dataset_config['lookback_window'])}_h{int(dataset_config['forecast_horizon'])}_in{int(input_size)}.pt"
    )
    model = train_or_load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        x_train=x_train,
        y_train=y_train,
        device=device,
        train_if_missing=bool(config["models"].get("train_if_missing", False)),
        epochs=int(config["models"].get("train_epochs", 1)),
    )
    cleanup_cuda(reset_memory_stats=config["benchmark"].get("reset_memory_stats_between_runs", True))

    if backend == "pytorch":
        metrics = run_pytorch_inference_benchmark(
            model=model,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            precision=precision,
            device=device,
            warmup_iters=int(config["benchmark"]["warmup_iters"]),
            timed_iters=int(config["benchmark"]["timed_iters"]),
        )
    elif backend == "onnxruntime":
        metrics = run_onnx_inference_benchmark()
    elif backend == "tensorrt":
        metrics = run_tensorrt_inference_benchmark()
    elif backend == "mlx":
        metrics = run_mlx_inference_benchmark()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    record = {
        "model": model_name,
        "backend": backend,
        "precision": precision,
        "batch_size": batch_size,
        "repeat_index": repeat_index,
        "parameter_count": int(parameter_count),
        "lookback_window": int(dataset_config["lookback_window"]),
        "forecast_horizon": int(dataset_config["forecast_horizon"]),
        "tickers": dataset_config["tickers"],
        **metrics,
        **get_hardware_metadata(),
    }
    cleanup_cuda(reset_memory_stats=config["benchmark"].get("reset_memory_stats_between_runs", True))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    record = run_single_config(config, args.model, args.backend, args.precision, args.batch_size, args.repeat_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
