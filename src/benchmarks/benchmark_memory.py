from __future__ import annotations

import torch


def current_peak_memory_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA memory benchmarking was requested, but CUDA is unavailable. No fallback is allowed.")
    return {
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }
