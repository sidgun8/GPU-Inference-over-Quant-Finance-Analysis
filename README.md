# Benchmarking Memory-Efficient Transformer Inference for Financial Time-Series Forecasting

This project benchmarks strict, reproducible inference configurations for financial time-series forecasting on consumer GPUs.

## Strict Research Semantics

Configured experiments do not silently fall back. If CUDA, a backend, a precision, W&B login, a model checkpoint, or an export path is required and unavailable, the benchmark fails visibly.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.benchmarks.run_all --config configs/benchmark.yaml
```

## Config

Edit `configs/benchmark.yaml` to choose datasets, models, backends, precisions, batch sizes, cleanup behavior, subprocess isolation, and W&B tracking.

## Outputs

- `results/raw`: JSON benchmark records
- `results/tables`: CSV summary tables
- `results/figures`: generated plots
- W&B runs if enabled
