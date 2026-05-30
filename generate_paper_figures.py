#!/usr/bin/env python3
"""Generate publication-quality figures for the GPU benchmarking paper."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Read aggregated data
csv_path = Path(r"C:\Users\sidsr\GPUs for Financial Engineering\aggregated_results_v2.csv")
output_dir = Path(r"C:\Users\sidsr\GPUs for Financial Engineering\results\latest\figures")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading aggregated data from {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Data shape: {df.shape}")
print(f"Models: {df['model'].unique()}")
print(f"Batch sizes: {df['batch_size'].unique()}")

# Model name mapping for display
model_names = {
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'informer': 'Informer',
    'tiny_time_mixer': 'Tiny Time Mixer'
}

df['model_display'] = df['model'].map(model_names)

# Color mapping
colors = {
    'fp32': '#1f77b4',
    'fp16': '#ff7f0e'
}

generated_files = []

# 1. Latency vs Batch Size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, model in enumerate(df['model'].unique()):
    ax = axes[idx]
    model_data = df[df['model'] == model]
    for precision in ['fp32', 'fp16']:
        prec_data = model_data[model_data['precision'] == precision]
        if len(prec_data) > 0:
            ax.plot(prec_data['batch_size'], prec_data['latency_mean_ms'], 
                   marker='o', label=f'{precision.upper()}', color=colors[precision], linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(model_names[model])
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.suptitle('Latency vs Batch Size by Model', fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / 'latency_vs_batch_size.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 2. Throughput vs Batch Size
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, model in enumerate(df['model'].unique()):
    ax = axes[idx]
    model_data = df[df['model'] == model]
    for precision in ['fp32', 'fp16']:
        prec_data = model_data[model_data['precision'] == precision]
        if len(prec_data) > 0:
            ax.plot(prec_data['batch_size'], prec_data['throughput_samples_per_sec'], 
                   marker='s', label=f'{precision.upper()}', color=colors[precision], linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title(model_names[model])
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.suptitle('Throughput vs Batch Size by Model', fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / 'throughput_vs_batch_size.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 3. GPU Utilization Comparison
fig, ax = plt.subplots(figsize=(10, 6))
batch_sizes = sorted(df['batch_size'].unique())
x = np.arange(len(batch_sizes))
width = 0.35

for i, model in enumerate(df['model'].unique()):
    model_data = df[df['model'] == model]
    fp32_util = model_data[model_data['precision'] == 'fp32']['gpu_utilization_percent_mean'].values
    fp16_util = model_data[model_data['precision'] == 'fp16']['gpu_utilization_percent_mean'].values
    
    offset = (i - 1.5) * width * 2
    ax.bar(x + offset, fp32_util, width, label=f'{model_names[model]} (FP32)', alpha=0.7)
    ax.bar(x + offset + width, fp16_util, width, label=f'{model_names[model]} (FP16)', alpha=0.7)

ax.set_xlabel('Batch Size')
ax.set_ylabel('GPU Utilization (%)')
ax.set_title('GPU Utilization Comparison')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
output_path = output_dir / 'gpu_utilization_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 4. GPU Power & Temperature
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (metric, ylabel, title) in enumerate([
    ('gpu_power_watts_mean', 'Power (W)', 'GPU Power Consumption'),
    ('gpu_temperature_c_mean', 'Temperature (°C)', 'GPU Temperature')
]):
    ax = axes[idx]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        for precision in ['fp32', 'fp16']:
            prec_data = model_data[model_data['precision'] == precision]
            if len(prec_data) > 0:
                ax.plot(prec_data['batch_size'], prec_data[metric], 
                       marker='o', label=f'{model_names[model]} ({precision.upper()})', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = output_dir / 'gpu_power_temperature.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 5. Accuracy Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
accuracy_metrics = ['mae', 'rmse', 'mape', 'directional_accuracy']
metric_labels = ['MAE', 'RMSE', 'MAPE (%)', 'Directional Accuracy']

for idx, (metric, label) in enumerate(zip(accuracy_metrics, metric_labels)):
    ax = axes[idx]
    x = np.arange(len(df['model'].unique()))
    width = 0.35
    
    for i, precision in enumerate(['fp32', 'fp16']):
        values = []
        for model in df['model'].unique():
            model_data = df[(df['model'] == model) & (df['precision'] == precision)]
            if len(model_data) > 0:
                values.append(model_data[metric].iloc[0])
        
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=f'{precision.upper()}', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(label)
    ax.set_title(f'{label} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([model_names[m] for m in df['model'].unique()])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Accuracy Metrics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / 'accuracy_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 6. Memory Usage
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df['model'].unique()))
width = 0.35

for i, precision in enumerate(['fp32', 'fp16']):
    alloc_values = []
    reserved_values = []
    for model in df['model'].unique():
        model_data = df[(df['model'] == model) & (df['precision'] == precision)]
        if len(model_data) > 0:
            alloc_values.append(model_data['peak_memory_allocated_mb'].iloc[0])
            reserved_values.append(model_data['peak_memory_reserved_mb'].iloc[0])
    
    offset = (i - 0.5) * width
    ax.bar(x + offset - width/2, alloc_values, width/2, label=f'{precision.upper()} Allocated', alpha=0.7)
    ax.bar(x + offset, reserved_values, width/2, label=f'{precision.upper()} Reserved', alpha=0.7)

ax.set_xlabel('Model')
ax.set_ylabel('Memory (MB)')
ax.set_title('Memory Usage Comparison')
ax.set_xticks(x)
ax.set_xticklabels([model_names[m] for m in df['model'].unique()])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
output_path = output_dir / 'memory_usage_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 7. Precision Impact (FP32 vs FP16)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
metrics = ['latency_mean_ms', 'throughput_samples_per_sec', 'gpu_utilization_percent_mean', 'peak_memory_allocated_mb']
metric_labels = ['Latency (ms)', 'Throughput (samples/s)', 'GPU Util (%)', 'Memory (MB)']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        fp32_data = model_data[model_data['precision'] == 'fp32']
        fp16_data = model_data[model_data['precision'] == 'fp16']
        
        if len(fp32_data) > 0 and len(fp16_data) > 0:
            ax.plot(fp32_data['batch_size'], fp32_data[metric], 
                   marker='o', linestyle='-', label=f'{model_names[model]} FP32', linewidth=2)
            ax.plot(fp16_data['batch_size'], fp16_data[metric], 
                   marker='s', linestyle='--', label=f'{model_names[model]} FP16', linewidth=2)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel(label)
    ax.set_title(f'{label}: FP32 vs FP16')
    ax.set_xscale('log')
    if metric == 'throughput_samples_per_sec':
        ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Precision Impact Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / 'precision_impact.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

# 8. Latency Distribution (Box Plot)
fig, ax = plt.subplots(figsize=(12, 6))
# Need to read original CSV for distribution data
original_csv = Path(r"C:\Users\sidsr\GPUs for Financial Engineering\wandb_export_2026-05-29T22_10_45.506+05_30.csv")
df_orig = pd.read_csv(original_csv)
df_orig['model'] = df_orig['Name'].str.extract(r'(\w+)-pytorch')[0]
df_orig['precision'] = df_orig['Name'].str.extract(r'-(fp\d+)-')[0]
df_orig['batch_size'] = df_orig['Name'].str.extract(r'-bs(\d+)-')[0].astype(int)
df_orig['model_display'] = df_orig['model'].map(model_names)

# Filter for batch size 32 for cleaner visualization
df_bs32 = df_orig[df_orig['batch_size'] == 32]

plot_data = []
labels = []
for model in df_bs32['model'].unique():
    for precision in ['fp32', 'fp16']:
        data = df_bs32[(df_bs32['model'] == model) & (df_bs32['precision'] == precision)]['latency_mean_ms'].values
        if len(data) > 0:
            plot_data.append(data)
            labels.append(f'{model_names[model]}\n{precision.upper()}')

bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], [colors['fp32'] if 'FP32' in label else colors['fp16'] for label in labels]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Latency (ms)')
ax.set_title('Latency Distribution by Model and Precision (Batch Size 32)')
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
output_path = output_dir / 'latency_distribution.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close()
generated_files.append(output_path)
print(f"Saved: {output_path}")

print(f"\n{'='*60}")
print(f"Generated {len(generated_files)} figures:")
for f in generated_files:
    print(f"  - {f}")
print(f"{'='*60}")
