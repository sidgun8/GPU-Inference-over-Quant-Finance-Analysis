from __future__ import annotations


def require_quantization_path(backend: str, precision: str) -> None:
    if precision == "int8":
        raise RuntimeError(f"INT8 is configured for backend={backend}, but a strict INT8 path has not been implemented yet.")
