from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.benchmarks.run_single_config import run_single_config
from src.tracking.local_logger import LocalLogger
from src.tracking.wandb_logger import WandbLogger
from src.utils.config import load_config
from src.utils.plotting import write_aggregated_summary, write_basic_plots, write_training_curve_plots
from src.utils.validation import validate_config


def build_matrix(config: dict[str, Any]) -> list[tuple[str, str, str, int, int]]:
    matrix: list[tuple[str, str, str, int, int]] = []
    for model in config["models"]["enabled"]:
        for backend in config["backends"]["enabled"]:
            for precision in config["precisions"]["enabled"]:
                for batch_size in config["batch_sizes"]:
                    for repeat_index in range(int(config["benchmark"].get("repeat_configs", 1))):
                        matrix.append((model, backend, precision, int(batch_size), repeat_index))
    return matrix


def run_subprocess_config(config_path: Path, output_path: Path, model: str, backend: str, precision: str, batch_size: int, repeat_index: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.benchmarks.run_single_config",
        "--config",
        str(config_path),
        "--model",
        model,
        "--backend",
        backend,
        "--precision",
        precision,
        "--batch-size",
        str(batch_size),
        "--repeat-index",
        str(repeat_index),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Configured benchmark subprocess failed. No fallback is allowed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-completed runs")
    parser.add_argument("--force", action="store_true", help="Force re-run all configs (ignores --resume)")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.model is not None:
        config["models"]["enabled"] = [args.model]
    if args.backend is not None:
        config["backends"]["enabled"] = [args.backend]
    if args.precision is not None:
        config["precisions"]["enabled"] = [args.precision]
    if args.batch_size is not None:
        config["batch_sizes"] = [args.batch_size]
    if args.wandb:
        config["tracking"]["wandb"]["enabled"] = True
    if args.wandb_project is not None:
        config["tracking"]["wandb"]["project"] = args.wandb_project

    validate_config(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    matrix_label = f"{'-'.join(config['models']['enabled'])}_{'-'.join(config['precisions']['enabled'])}_bs{'-'.join(str(value) for value in config['batch_sizes'])}"
    
    # Auto-detect previous run for resume mode
    if args.resume and not args.run_name:
        results_runs_dir = Path(config["tracking"]["local"]["results_dir"]) / "runs"
        if results_runs_dir.exists():
            existing_runs = [d for d in results_runs_dir.iterdir() if d.is_dir() and matrix_label in d.name]
            if existing_runs:
                # Sort by modification time, get most recent that actually has data
                existing_runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                run_group = None
                for run_dir in existing_runs:
                    raw_dir = run_dir / "raw"
                    if raw_dir.exists() and any(raw_dir.glob("*.json")):
                        run_group = run_dir.name
                        print(f"Resuming from previous run: {run_group}")
                        break
                if run_group is None:
                    # No valid run found with data
                    run_group = f"{timestamp}_{matrix_label}"
                    print(f"Previous run directory exists but has no data, starting fresh: {run_group}")
            else:
                run_group = f"{timestamp}_{matrix_label}"
                print(f"No previous run found, starting fresh: {run_group}")
        else:
            run_group = f"{timestamp}_{matrix_label}"
    else:
        run_group = args.run_name or f"{timestamp}_{matrix_label}"
    
    local_logger = LocalLogger(config["tracking"]["local"]["results_dir"], run_id=run_group)
    wandb_logger = WandbLogger(config["tracking"]["wandb"], group=run_group)
    matrix = build_matrix(config)
    subprocess_mode = bool(config["benchmark"].get("run_each_config_in_subprocess", True))
    fail_fast = bool(config["benchmark"].get("fail_fast", True))
    resume_mode = args.resume and not args.force

    for model, backend, precision, batch_size, repeat_index in matrix:
        temp_output = local_logger.raw_dir / f"subprocess_{model}_{backend}_{precision}_bs{batch_size}_rep{repeat_index}.json"
        final_output = local_logger.raw_dir / f"{run_group}_{model}_{backend}_{precision}_bs{batch_size}_rep{repeat_index}.json"
        
        # Check if already completed in resume mode (check final logged file)
        if resume_mode and final_output.exists():
            print(f"Skipping {model}-{backend}-{precision}-bs{batch_size}-rep{repeat_index} (already completed)")
            with final_output.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            local_logger.records.append(record)
            wandb_logger.log_record(record)
            continue
        
        try:
            if subprocess_mode:
                record = run_subprocess_config(config_path, temp_output, model, backend, precision, batch_size, repeat_index)
            else:
                record = run_single_config(config, model, backend, precision, batch_size, repeat_index)
            local_logger.log_record(record)
            wandb_logger.log_record(record)
        except Exception:
            if fail_fast:
                raise
            raise RuntimeError("fail_fast=false is not supported for strict academic benchmarking in this implementation.")

    summary_path = local_logger.write_summary()
    aggregated_summary_path = write_aggregated_summary(summary_path, local_logger.tables_dir)
    figure_paths = [
        *write_basic_plots(summary_path, local_logger.figures_dir),
        *write_training_curve_plots(summary_path, local_logger.figures_dir),
    ]
    wandb_logger.log_artifacts([summary_path, aggregated_summary_path, *figure_paths])
    print(f"Benchmark complete. Summary written to {summary_path}")
    print(f"Aggregated summary written to {aggregated_summary_path}")
    print(f"Latest copy written to {local_logger.latest_dir / 'tables' / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
