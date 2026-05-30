import pandas as pd
import numpy as np

# Read the W&B export CSV
csv_path = r"C:\Users\sidsr\GPUs for Financial Engineering\wandb_export_2026-05-29T21_40_09.109+05_30.csv"
df = pd.read_csv(csv_path)

# Filter out artifact rows
df = df[df['Name'].str.contains('artifacts') == False]

# Extract model, precision, batch_size, repeat_index from Name
df['model'] = df['Name'].str.extract(r'^(lstm|transformer|tiny_time_mixer|informer)')[0]
df['precision'] = df['Name'].str.extract(r'(fp32|fp16)')[0]
df['batch_size'] = df['Name'].str.extract(r'bs(\d+)')[0].astype(int)
df['repeat_index'] = df['Name'].str.extract(r'rep(\d+)')[0].astype(int)

# Group by model, precision, batch_size and aggregate
group_cols = ['model', 'precision', 'batch_size']
agg_cols = {
    'latency_mean_ms': 'mean',
    'latency_median_ms': 'mean',
    'latency_p95_ms': 'mean',
    'latency_p99_ms': 'mean',
    'latency_min_ms': 'mean',
    'latency_max_ms': 'mean',
    'throughput_samples_per_sec': 'mean',
    'peak_memory_allocated_mb': 'mean',
    'peak_memory_reserved_mb': 'mean',
    'mae': 'mean',
    'rmse': 'mean',
    'mape': 'mean',
    'directional_accuracy': 'mean',
    'gpu_utilization_percent_mean': 'mean',
    'gpu_power_watts_mean': 'mean',
    'gpu_temperature_c_mean': 'mean'
}

aggregated = df.groupby(group_cols).agg(agg_cols).reset_index()

# Sort by model, precision, batch_size
aggregated = aggregated.sort_values(['model', 'precision', 'batch_size'])

print("Aggregated Results:")
print(aggregated[['model', 'precision', 'batch_size', 'latency_median_ms', 'latency_p95_ms', 'throughput_samples_per_sec', 'peak_memory_allocated_mb', 'mae', 'rmse', 'mape', 'directional_accuracy']])

# Save to CSV for reference
aggregated.to_csv(r"C:\Users\sidsr\GPUs for Financial Engineering\aggregated_results.csv", index=False)
print("\nSaved to aggregated_results.csv")

# Print LaTeX table for latency and throughput
print("\n=== Latency and Throughput Table ===")
for model in ['lstm', 'transformer', 'tiny_time_mixer', 'informer']:
    for precision in ['fp32', 'fp16']:
        for bs in [1, 8, 32]:
            row = aggregated[(aggregated['model'] == model) & 
                           (aggregated['precision'] == precision) & 
                           (aggregated['batch_size'] == bs)]
            if not row.empty:
                r = row.iloc[0]
                print(f"{model} & {precision} & {bs} & {r['latency_median_ms']:.3f} & {r['latency_p95_ms']:.3f} & {r['throughput_samples_per_sec']:.1f} \\\\")

# Print LaTeX table for memory
print("\n=== GPU Memory Usage Table ===")
for model in ['lstm', 'transformer', 'tiny_time_mixer', 'informer']:
    for precision in ['fp32', 'fp16']:
        for bs in [1, 8, 32]:
            row = aggregated[(aggregated['model'] == model) & 
                           (aggregated['precision'] == precision) & 
                           (aggregated['batch_size'] == bs)]
            if not row.empty:
                r = row.iloc[0]
                print(f"{model} & {precision} & {bs} & {r['peak_memory_allocated_mb']:.1f} & {r['peak_memory_reserved_mb']:.1f} \\\\")

# Print LaTeX table for accuracy (unique per model/precision)
print("\n=== Forecasting Accuracy Table ===")
for model in ['lstm', 'transformer', 'tiny_time_mixer', 'informer']:
    for precision in ['fp32', 'fp16']:
        row = aggregated[(aggregated['model'] == model) & (aggregated['precision'] == precision)]
        if not row.empty:
            r = row.iloc[0]
            print(f"{model} & {precision} & {r['mae']:.4f} & {r['rmse']:.4f} & {r['mape']:.2f} & {r['directional_accuracy']:.4f} \\\\")
