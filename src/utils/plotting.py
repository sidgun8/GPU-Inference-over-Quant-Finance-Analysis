from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Set style for paper-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


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


def write_aggregated_summary(table_path: str | Path, output_dir: str | Path) -> Path:
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table_path)
    group_columns = ["model", "backend", "precision", "batch_size"]
    metric_columns = [
        "latency_median_ms",
        "latency_p95_ms",
        "throughput_samples_per_sec",
        "peak_memory_allocated_mb",
        "mae",
        "rmse",
        "mape",
        "directional_accuracy",
    ]
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n"] = int(len(group))
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    output_path = output_dir / "benchmark_aggregated_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def write_training_curve_plots(table_path: str | Path, output_dir: str | Path) -> list[Path]:
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table_path)
    if "training_history_path" not in df.columns:
        return []
    histories = []
    seen_paths = set()
    for _, row in df.dropna(subset=["training_history_path"]).iterrows():
        history_path = Path(row["training_history_path"])
        if not history_path.exists() or history_path in seen_paths:
            continue
        seen_paths.add(history_path)
        with history_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for point in payload.get("history", []):
            histories.append({
                "model": row["model"],
                "epoch": point["epoch"],
                "train_loss": point["train_loss"],
                "val_loss": point["val_loss"],
            })
    if not histories:
        return []
    history_df = pd.DataFrame(histories)
    long_df = history_df.melt(
        id_vars=["model", "epoch"],
        value_vars=["train_loss", "val_loss"],
        var_name="split",
        value_name="loss",
    )
    path = output_dir / "training_validation_loss_curves.png"
    grid = sns.relplot(
        data=long_df,
        x="epoch",
        y="loss",
        hue="split",
        col="model",
        col_wrap=2,
        kind="line",
        facet_kws={"sharey": False},
    )
    grid.fig.suptitle("Training and validation loss by model", y=1.02)
    grid.fig.tight_layout()
    grid.fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)
    return [path]


def write_paper_figures(table_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate paper-ready figures for the research paper."""
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table_path)
    created: list[Path] = []
    if df.empty:
        return created

    # Color palette for models (use original names for mapping)
    model_colors = {
        'tiny_time_mixer': '#2E86AB',
        'informer': '#A23B72',
        'lstm': '#F18F01',
        'transformer': '#C73E1D'
    }
    
    # Function to get color for model name
    def get_model_color(model_name):
        original_name = model_name.lower().replace(' ', '_')
        return model_colors.get(original_name, '#666666')
    
    # Precision colors (uppercase to match data)
    precision_colors = {
        'FP32': '#3B82F6',
        'FP16': '#EF4444'
    }

    # Figure 1: Median latency by model and batch size (FP32)
    df_fp32 = df[df['precision'] == 'fp32'].copy()
    if not df_fp32.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Aggregate by model and batch size, taking mean of repeats
        agg_df = df_fp32.groupby(['model', 'batch_size'])['latency_median_ms'].mean().reset_index()
        
        sns.barplot(data=agg_df, x='batch_size', y='latency_median_ms', hue='model', 
                    palette=model_colors, ax=ax, width=0.7)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Median Latency (ms)', fontweight='bold')
        ax.set_title('Median Inference Latency by Model and Batch Size (FP32)', 
                    fontweight='bold', pad=15)
        # Customize legend labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.replace('_', ' ').title() for label in labels]
        ax.legend(handles, new_labels, title='Model', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        path = output_dir / 'latency_median_fp32.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created.append(path)

    # Figure 2: Median latency by model and batch size (FP16)
    df_fp16 = df[df['precision'] == 'fp16'].copy()
    if not df_fp16.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Aggregate by model and batch size, taking mean of repeats
        agg_df = df_fp16.groupby(['model', 'batch_size'])['latency_median_ms'].mean().reset_index()
        
        sns.barplot(data=agg_df, x='batch_size', y='latency_median_ms', hue='model', 
                    palette=model_colors, ax=ax, width=0.7)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Median Latency (ms)', fontweight='bold')
        ax.set_title('Median Inference Latency by Model and Batch Size (FP16)', 
                    fontweight='bold', pad=15)
        # Customize legend labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.replace('_', ' ').title() for label in labels]
        ax.legend(handles, new_labels, title='Model', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        path = output_dir / 'latency_median_fp16.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created.append(path)

    # Figure 3: Throughput-memory tradeoff at batch size 32
    df_bs32 = df[df['batch_size'] == 32].copy()
    if not df_bs32.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aggregate by model and precision
        agg_df = df_bs32.groupby(['model', 'precision']).agg({
            'peak_memory_allocated_mb': 'mean',
            'throughput_samples_per_sec': 'mean'
        }).reset_index()
        
        for _, row in agg_df.iterrows():
            model_name = row['model'].replace('_', ' ').title()
            precision = row['precision'].upper()
            color = model_colors.get(row['model'], '#666666')
            marker = 'o' if row['precision'] == 'fp32' else 's'
            
            ax.scatter(row['peak_memory_allocated_mb'], row['throughput_samples_per_sec'],
                      label=f'{model_name} ({precision})', s=200, alpha=0.8,
                      color=color, marker=marker, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Peak Allocated GPU Memory (MB)', fontweight='bold')
        ax.set_ylabel('Throughput (samples/sec)', fontweight='bold')
        ax.set_title('Throughput-Memory Tradeoff at Batch Size 32', 
                    fontweight='bold', pad=15)
        ax.legend(title='Model (Precision)', frameon=True, shadow=True, 
                 bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        plt.tight_layout()
        path = output_dir / 'throughput_memory_pareto_bs32.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created.append(path)

    # Figure 4: Accuracy metrics by model and precision
    accuracy_cols = ['mae', 'rmse', 'mape', 'directional_accuracy']
    accuracy_data = df[['model', 'precision'] + accuracy_cols].drop_duplicates()
    if not accuracy_data.empty:
        # Keep original model names for grouping, create display names
        accuracy_data['model_display'] = accuracy_data['model'].str.replace('_', ' ').str.title()
        accuracy_data['precision'] = accuracy_data['precision'].str.upper()
        
        # Melt for plotting
        melted = accuracy_data.melt(
            id_vars=['model', 'model_display', 'precision'],
            value_vars=accuracy_cols,
            var_name='metric',
            value_name='value'
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metric_labels = {
            'mae': 'MAE',
            'rmse': 'RMSE',
            'mape': 'MAPE (%)',
            'directional_accuracy': 'Directional Accuracy'
        }
        
        for idx, metric in enumerate(accuracy_cols):
            ax = axes[idx]
            metric_data = melted[melted['metric'] == metric]
            sns.barplot(data=metric_data, x='model_display', y='value', hue='precision', 
                       palette=precision_colors, ax=ax, width=0.6)
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontweight='bold')
            ax.set_title(f'{metric_labels[metric]} by Model and Precision', 
                        fontweight='bold', pad=10)
            ax.legend(title='Precision', frameon=True, shadow=True)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Forecasting Accuracy Metrics', fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        path = output_dir / 'accuracy_metrics.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created.append(path)

    return created
