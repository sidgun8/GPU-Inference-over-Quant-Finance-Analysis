from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def write_basic_plots(table_path: str | Path, output_dir: str | Path) -> list[Path]:
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table_path)
    created: list[Path] = []
    if df.empty:
        return created

    latency_path = output_dir / "latency_by_config.png"
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="batch_size", y="latency_median_ms", hue="model")
    plt.title("Median latency by model and batch size")
    plt.tight_layout()
    plt.savefig(latency_path, dpi=200)
    plt.close()
    created.append(latency_path)

    memory_path = output_dir / "memory_by_config.png"
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="batch_size", y="peak_memory_allocated_mb", hue="model")
    plt.title("Peak allocated GPU memory by model and batch size")
    plt.tight_layout()
    plt.savefig(memory_path, dpi=200)
    plt.close()
    created.append(memory_path)
    return created
