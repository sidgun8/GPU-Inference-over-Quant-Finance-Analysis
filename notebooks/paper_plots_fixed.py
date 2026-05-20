from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300

cwd = Path.cwd()
relative_summary = Path("results") / "runs" / "full-pytorch-lb252-h21-fixed-v3" / "tables" / "benchmark_summary.csv"
if (cwd / relative_summary).exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / relative_summary).exists():
    PROJECT_ROOT = cwd.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "full-pytorch-lb252-h21-fixed-v3"
SUMMARY_CSV = RUN_DIR / "tables" / "benchmark_summary.csv"
FIGURE_DIR = RUN_DIR / "paper_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV).dropna(how="all")
df["model_label"] = df["model"].replace({
    "lstm": "LSTM",
    "transformer": "Transformer",
    "tiny_time_mixer": "Tiny Time Mixer",
    "informer": "Informer",
})
df["precision_label"] = df["precision"].str.upper()

summary = (
    df.groupby(["model", "model_label", "precision", "precision_label", "batch_size"], as_index=False)
    .agg(
        repeats=("repeat_index", "count"),
        latency_median_ms=("latency_median_ms", "mean"),
        latency_p95_ms=("latency_p95_ms", "mean"),
        throughput_samples_per_sec=("throughput_samples_per_sec", "mean"),
        peak_memory_allocated_mb=("peak_memory_allocated_mb", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        mape=("mape", "mean"),
        directional_accuracy=("directional_accuracy", "mean"),
        parameter_count=("parameter_count", "mean"),
    )
)
summary.to_csv(FIGURE_DIR / "paper_aggregated_summary.csv", index=False)

def save_barplot(data, x, y, hue, title, ylabel, filename, log_y=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

for precision in ["fp32", "fp16"]:
    subset = summary[summary["precision"] == precision]
    save_barplot(subset, "batch_size", "latency_median_ms", "model_label", f"Median latency by model and batch size ({precision.upper()})", "Median latency (ms)", f"latency_median_{precision}.png")
    save_barplot(subset, "batch_size", "throughput_samples_per_sec", "model_label", f"Throughput by model and batch size ({precision.upper()})", "Throughput (samples/s)", f"throughput_{precision}.png", log_y=True)
    save_barplot(subset, "batch_size", "peak_memory_allocated_mb", "model_label", f"Peak allocated GPU memory ({precision.upper()})", "Peak allocated memory (MB)", f"memory_{precision}.png")

accuracy = summary[summary["batch_size"] == 1].copy()
accuracy_metrics = accuracy.melt(
    id_vars=["model_label", "precision_label"],
    value_vars=["mae", "rmse", "mape", "directional_accuracy"],
    var_name="metric",
    value_name="value",
)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, metric in zip(axes.ravel(), ["mae", "rmse", "mape", "directional_accuracy"]):
    subset = accuracy_metrics[accuracy_metrics["metric"] == metric]
    sns.barplot(data=subset, x="model_label", y="value", hue="precision_label", errorbar=None, ax=ax)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Precision")
fig.suptitle("Forecasting accuracy metrics by model and precision", y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "accuracy_metrics.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
pareto = summary[summary["batch_size"] == 32].copy()
sns.scatterplot(
    data=pareto,
    x="peak_memory_allocated_mb",
    y="throughput_samples_per_sec",
    hue="model_label",
    style="precision_label",
    s=140,
    ax=ax,
)
for _, row in pareto.iterrows():
    ax.annotate(row["precision_label"], (row["peak_memory_allocated_mb"], row["throughput_samples_per_sec"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_title("Throughput-memory tradeoff at batch size 32")
ax.set_xlabel("Peak allocated memory (MB)")
ax.set_ylabel("Throughput (samples/s)")
ax.set_yscale("log")
ax.legend(title="Model / Precision", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "throughput_memory_pareto_bs32.png", bbox_inches="tight")
plt.close(fig)

print("Created figures:")
for path in sorted(FIGURE_DIR.glob("*.png")):
    print(path)
