from __future__ import annotations

import platform
from typing import Any

import psutil
import torch


def require_cuda_device(allow_cpu_fallback: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu_fallback:
        return torch.device("cpu")
    raise RuntimeError("CUDA is required by config, but no CUDA device is available. No CPU fallback is allowed.")


def get_hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 3),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 3),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_device_index": index,
            }
        )
    return metadata
