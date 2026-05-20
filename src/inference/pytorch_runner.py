from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from src.utils.cleanup import synchronize_if_cuda
from src.utils.gpu_telemetry import GpuTelemetrySampler
from src.utils.metrics import latency_summary, regression_metrics


def _apply_precision(model: nn.Module, inputs: torch.Tensor, precision: str) -> tuple[nn.Module, torch.Tensor]:
    if precision == "fp32":
        return model.float(), inputs.float()
    if precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 was configured, but CUDA is unavailable. No fallback is allowed.")
        return model.half(), inputs.half()
    if precision == "int8":
        raise RuntimeError("INT8 PyTorch GPU path is not implemented yet. Remove int8 from config or implement a strict quantized backend.")
    raise ValueError(f"Unsupported precision: {precision}")


def run_pytorch_inference_benchmark(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    precision: str,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
    eval_batch_size: int = 256,
) -> dict[str, Any]:
    model = model.to(device).eval()
    sample_count = min(batch_size, len(x_test))
    if sample_count < batch_size:
        raise RuntimeError(f"Configured batch_size={batch_size} exceeds available test samples={len(x_test)}. No fallback is allowed.")
    batch_np = x_test[:sample_count]
    inputs = torch.from_numpy(batch_np).to(device)
    model, inputs = _apply_precision(model, inputs, precision)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(inputs)
        synchronize_if_cuda()

        timings: list[float] = []
        outputs = None
        telemetry_device_index = device.index if device.index is not None else torch.cuda.current_device()
        with GpuTelemetrySampler(device_index=telemetry_device_index) as telemetry:
            for _ in range(timed_iters):
                synchronize_if_cuda()
                start = time.perf_counter()
                outputs = model(inputs)
                synchronize_if_cuda()
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)
        telemetry_summary = telemetry.summary()

    if outputs is None:
        raise RuntimeError("No inference outputs were produced.")
    eval_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, len(x_test), eval_batch_size):
            eval_batch = torch.from_numpy(x_test[start_index:start_index + eval_batch_size]).to(device)
            if precision == "fp16":
                eval_batch = eval_batch.half()
            else:
                eval_batch = eval_batch.float()
            eval_outputs = model(eval_batch)
            eval_predictions.append(eval_outputs.float().detach().cpu().numpy())
    pred_np = np.concatenate(eval_predictions, axis=0)
    metrics = regression_metrics(y_test, pred_np)
    summary = latency_summary(timings)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
    throughput = batch_size / (summary["latency_median_ms"] / 1000.0)
    return {
        **summary,
        **metrics,
        "throughput_samples_per_sec": float(throughput),
        "peak_memory_allocated_mb": float(peak_allocated),
        "peak_memory_reserved_mb": float(peak_reserved),
        "accuracy_eval_samples": int(len(y_test)),
        **telemetry_summary,
    }
