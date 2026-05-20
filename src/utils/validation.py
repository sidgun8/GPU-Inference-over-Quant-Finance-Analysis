from __future__ import annotations

import importlib.util
from typing import Any

import torch


SUPPORTED_MODELS = {"lstm", "transformer", "tiny_time_mixer", "informer"}
SUPPORTED_BACKENDS = {"pytorch", "onnxruntime", "tensorrt", "mlx"}
SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}


def _require_package(package_name: str, label: str) -> None:
    if importlib.util.find_spec(package_name) is None:
        raise RuntimeError(f"Configured backend/dependency is unavailable: {label}. No fallback is allowed.")


def validate_config(config: dict[str, Any]) -> None:
    models = set(config["models"]["enabled"])
    backends = set(config["backends"]["enabled"])
    precisions = set(config["precisions"]["enabled"])

    unknown_models = models - SUPPORTED_MODELS
    unknown_backends = backends - SUPPORTED_BACKENDS
    unknown_precisions = precisions - SUPPORTED_PRECISIONS
    if unknown_models:
        raise ValueError(f"Unsupported configured models: {sorted(unknown_models)}")
    if unknown_backends:
        raise ValueError(f"Unsupported configured backends: {sorted(unknown_backends)}")
    if unknown_precisions:
        raise ValueError(f"Unsupported configured precisions: {sorted(unknown_precisions)}")

    if config["backends"].get("allow_fallback", False):
        raise ValueError("allow_fallback must be false for strict academic benchmarking.")
    if config["precisions"].get("allow_precision_fallback", False):
        raise ValueError("allow_precision_fallback must be false for strict academic benchmarking.")
    if config["hardware"].get("allow_cpu_fallback", False):
        raise ValueError("allow_cpu_fallback must be false for strict academic benchmarking.")

    if config["hardware"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by config but unavailable. No fallback is allowed.")
    if "fp16" in precisions and not torch.cuda.is_available():
        raise RuntimeError("FP16 GPU benchmarking requires CUDA. No fallback is allowed.")
    if "onnxruntime" in backends:
        _require_package("onnxruntime", "ONNX Runtime")
    if "tensorrt" in backends:
        _require_package("tensorrt", "TensorRT")
    if "mlx" in backends:
        _require_package("mlx", "MLX")
    if config.get("tracking", {}).get("wandb", {}).get("enabled", False):
        _require_package("wandb", "Weights & Biases")
