from __future__ import annotations

import gc

import torch


def cleanup_cuda(reset_memory_stats: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if reset_memory_stats:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
