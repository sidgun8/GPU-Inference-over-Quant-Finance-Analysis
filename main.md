::: IEEEkeywords
financial forecasting, GPU inference, time-series forecasting, deep
learning, LSTM, Transformer, Informer, Tiny Time Mixer, PyTorch, CUDA,
benchmarking
:::

# Introduction

Financial time-series forecasting is an important task in quantitative
finance, portfolio management, algorithmic trading, and risk analysis.
Modern forecasting pipelines increasingly rely on deep learning models
capable of learning temporal dependencies from historical market data.
However, the practical deployment of these models is constrained not
only by predictive accuracy but also by inference latency, memory
consumption, throughput, and hardware availability.

In real-world financial systems, inference may need to be performed
repeatedly across many assets, rolling windows, or trading strategies.
Therefore, small improvements in inference latency or GPU memory usage
can significantly affect deployment cost and scalability. While high-end
data-center GPUs are often used for training and large-scale inference,
many academic and low-cost financial engineering workflows rely on
consumer GPUs. This motivates a systematic benchmark of financial
forecasting inference efficiency on accessible hardware.

This work develops and evaluates a strict benchmarking framework for
GPU-based financial time-series inference. The framework is designed
around reproducibility and experimental integrity. Unlike benchmarking
scripts that silently fall back to CPU execution or lower precision when
a configuration fails, this framework uses no-fallback semantics: if
CUDA, a selected backend, or a precision mode is unavailable, the
experiment fails rather than being replaced by an unintended
configuration.

The major contributions of this work are:

- A reproducible benchmark pipeline for financial time-series
  forecasting inference using GPU acceleration.

- A comparison framework for multiple neural architectures including
  LSTM, Transformer, Tiny Time Mixer, and Informer.

- A strict no-fallback benchmarking design that enforces configured
  device, backend, and precision requirements.

- Empirical results on consumer GPU hardware using PyTorch CUDA
  inference across 25 financial instruments.

- A paper-ready structure that can be extended with future results from
  additional models, precisions, backends, and hardware platforms.

# Related Work

Deep learning has become a widely used approach for financial
time-series forecasting. Recurrent neural networks, especially Long
Short-Term Memory networks, have historically been used due to their
ability to model sequential dependencies. More recently,
Transformer-based architectures have gained popularity because of their
attention mechanisms and parallelizable structure.

Informer and related efficient attention models have been proposed for
long-sequence time-series forecasting, reducing the computational burden
of standard self-attention mechanisms. Tiny Time Mixer and other compact
time-series models aim to improve deployment efficiency by reducing
parameter count and computational cost.

Most financial forecasting research emphasizes predictive metrics such
as mean absolute error, root mean squared error, or directional
accuracy. In contrast, deployment-oriented metrics such as inference
latency, GPU memory usage, and throughput are often underreported. This
paper addresses that gap by focusing on both forecasting quality and
inference efficiency.

# Methodology

## Dataset

The benchmark uses financial market data obtained through Yahoo Finance.
The current configuration includes 25 financial instruments spanning US
and Indian markets:

- US Equities: AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, JPM, GS, BAC,
  XOM, CVX, TSM

- ETFs: SPY, QQQ, DIA, IWM, SMH, CQQQ, FXI, VIXY

- Indian Indices: `^NSEI`, RELIANCE.NS, TCS.NS, INFY.NS

The dataset starts from January 1, 2015 and extends to the most recent
available date at execution time. The input features are:

- Open price

- High price

- Low price

- Close price

- Volume

The forecasting target is the closing price. The benchmark uses a
lookback window of 252 trading days and a forecast horizon of 21 trading
days. The train-test split is 80--20.

## Supervised Forecasting Formulation

Let $X_t \in \mathbb{R}^{L \times F}$ denote a historical sequence of
financial features, where $L$ is the lookback window and $F$ is the
number of input features. The model predicts a future target value:

$$\begin{equation}
    \hat{y}_{t+h} = f_{\theta}(X_t),
\end{equation}$$

where $h$ is the forecast horizon and $f_{\theta}$ is a neural
forecasting model with parameters $\theta$.

In the current benchmark:

$$\begin{equation}
    L = 252, \quad h = 21, \quad F = 5.
\end{equation}$$

## Models

The implementation supports four forecasting architectures.

### LSTM

The LSTM model is a recurrent neural network designed to capture
temporal dependencies through gated memory cells. It is commonly used
for sequential forecasting tasks and serves as the initial baseline in
the current experiments.

### Transformer

The Transformer model uses self-attention to model relationships across
time steps. Its parallel structure can improve computational efficiency
compared to recurrent models, especially for larger batch sizes and
longer sequences.

### Tiny Time Mixer

Tiny Time Mixer is included as a compact time-series architecture
intended for efficient inference. It is useful for studying the tradeoff
between model size, latency, and forecasting accuracy.

### Informer

Informer is an efficient Transformer-style architecture designed for
long-sequence time-series forecasting. It is included to evaluate
whether efficient attention mechanisms provide practical inference
benefits for financial workloads.

## Benchmark Configuration

The experimental configuration is summarized in
Table [1](#tab:config){reference-type="ref" reference="tab:config"}.

::: {#tab:config}
  **Component**       **Configuration**
  ------------------- ----------------------------------------------------
  Data source         Yahoo Finance
  Tickers             25 instruments (US equities, ETFs, Indian indices)
  Start date          2015-01-01
  Features            Open, High, Low, Close, Volume
  Target              Close
  Lookback window     252 trading days
  Forecast horizon    21 trading days
  Train split         0.8
  Models              LSTM, Transformer, Tiny Time Mixer, Informer
  Backend             PyTorch
  Precisions          FP32, FP16
  Batch sizes         1, 8, 32
  Warmup iterations   20
  Timed iterations    100
  Repeats             3
  Seed                42

  : Benchmark Configuration
:::

[]{#tab:config label="tab:config"}

## Hardware and Software Environment

The current available benchmark results were collected on the hardware
and software environment shown in
Table [2](#tab:hardware){reference-type="ref" reference="tab:hardware"}.

::: {#tab:hardware}
  **Component**            **Value**
  ------------------------ ------------------------------------
  Operating system         Windows 11
  CPU                      AMD64 Family 25 Model 116
  GPU                      NVIDIA GeForce RTX 4070 Laptop GPU
  GPU memory               8 GB
  CUDA available           Yes
  CUDA version             12.6
  GPU compute capability   8.9
  Python version           3.12.10
  PyTorch version          2.12.0+cu126
  System RAM               15.288 GB

  : Current Hardware and Software Environment
:::

[]{#tab:hardware label="tab:hardware"}

## Strict No-Fallback Benchmarking

A key design principle of the benchmark is strict experimental
integrity. The configuration requires CUDA execution and disables CPU
fallback. Precision fallback is also disabled. Therefore, if a requested
configuration cannot run exactly as specified, the benchmark reports a
failure rather than silently substituting another backend, device, or
precision.

This design is important for academic benchmarking because silent
fallback can produce misleading results. For example, if an FP16 CUDA
configuration silently falls back to FP32 CPU execution, the reported
latency and memory measurements would not correspond to the intended
experiment.

## Evaluation Metrics

The benchmark measures both forecasting quality and inference
efficiency.

### Forecasting Metrics

Mean Absolute Error is defined as:

$$\begin{equation}
    MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|.
\end{equation}$$

Root Mean Squared Error is defined as:

$$\begin{equation}
    RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}.
\end{equation}$$

Mean Absolute Percentage Error is defined as:

$$\begin{equation}
    MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{\max(|y_i|, \epsilon)}\right|.
\end{equation}$$

Directional accuracy is computed by comparing the sign of the predicted
and actual changes:

$$\begin{equation}
    DA = \frac{1}{n-1}\sum_{i=2}^{n}\mathbb{I}
    \left[
    \text{sign}(y_i-y_{i-1}) =
    \text{sign}(\hat{y}_i-\hat{y}_{i-1})
    \right].
\end{equation}$$

### Inference Efficiency Metrics

The benchmark records:

- Mean latency in milliseconds

- Median latency in milliseconds

- 95th percentile latency

- 99th percentile latency

- Minimum and maximum latency

- Throughput in samples per second

- Peak GPU memory allocated

- Peak GPU memory reserved

- GPU utilization telemetry

- GPU power and temperature telemetry

# Implementation

The benchmark is implemented in Python using PyTorch and CUDA. The
project is organized into modular components for data loading,
preprocessing, model construction, training, inference, metric
calculation, logging, and plotting.

## Data Pipeline

Historical market data is fetched using Yahoo Finance. The preprocessing
module converts raw time-series data into supervised learning arrays
using a sliding lookback window. Each input sample contains 252
historical trading days and five market features. The target corresponds
to the future closing price at the configured forecast horizon.

## Model Construction

A model factory constructs the requested forecasting model based on a
configuration name. The supported models are:

- `lstm`

- `transformer`

- `tiny_time_mixer`

- `informer`

All models use a hidden size of 64 in the current configuration.

## Training and Checkpointing

The training module either loads an existing checkpoint or trains the
model if the checkpoint is missing and training is enabled. The current
configuration allows training if missing and uses 30 training epochs.
The optimizer is AdamW with a learning rate of $10^{-3}$, and the loss
function is mean squared error.

## Inference Benchmarking

For each configuration, the benchmark executes a fixed number of warmup
iterations followed by timed iterations. Warmup iterations reduce
measurement noise from initialization overhead. Timed iterations are
used to compute latency summaries and throughput.

The experiment matrix is constructed from:

$$\begin{equation}
    \mathcal{M} \times \mathcal{B} \times \mathcal{P} \times \mathcal{S} \times \mathcal{R},
\end{equation}$$

where $\mathcal{M}$ is the model set, $\mathcal{B}$ is the backend set,
$\mathcal{P}$ is the precision set, $\mathcal{S}$ is the batch-size set,
and $\mathcal{R}$ is the repeat index set.

## Logging and Result Generation

Each benchmark configuration produces a raw JSON record containing model
information, hardware metadata, latency metrics, memory metrics,
accuracy metrics, and software versions. A summary CSV table is
generated after the benchmark completes. Basic plots are generated for
latency and memory comparisons.

The current plotting utilities generate:

- Median latency by model and batch size

- Peak allocated GPU memory by model and batch size

# Experimental Results

The completed PyTorch benchmark evaluates four forecasting
architectures: LSTM, Transformer, Tiny Time Mixer, and Informer. Each
model was tested under FP32 and FP16 precision at batch sizes 1, 8, and
32, with three repeated runs per configuration. Results were collected
on an NVIDIA GeForce RTX 4070 Laptop GPU using CUDA 12.6 and PyTorch
2.12.0. The benchmark reports both inference-efficiency metrics and
forecasting-quality metrics.

## Latency and Throughput

Table [\[tab:latency_results\]](#tab:latency_results){reference-type="ref"
reference="tab:latency_results"} reports the mean median latency, 95th
percentile latency, and throughput across three repeated runs for each
model, precision, and batch size.

::: table*
[]{#tab:latency_results label="tab:latency_results"}
:::

## GPU Memory Usage

Table [\[tab:memory_results\]](#tab:memory_results){reference-type="ref"
reference="tab:memory_results"} reports peak allocated GPU memory for
each model, precision, and batch size. Tiny Time Mixer uses the least
memory across the evaluated architectures, while Informer requires the
most memory due to its larger parameter count.

::: table*
[]{#tab:memory_results label="tab:memory_results"}
:::

## Forecasting Accuracy

Table [\[tab:forecasting_accuracy_results\]](#tab:forecasting_accuracy_results){reference-type="ref"
reference="tab:forecasting_accuracy_results"} reports
forecasting-quality metrics for each model and precision. Since the same
trained checkpoint is evaluated across batch sizes, the forecasting
metrics are constant across batch-size configurations for a given model
and precision.

::: table*
[]{#tab:forecasting_accuracy_results
label="tab:forecasting_accuracy_results"}
:::

## Combined Efficiency and Accuracy

Table [\[tab:combined_bs32_results\]](#tab:combined_bs32_results){reference-type="ref"
reference="tab:combined_bs32_results"} summarizes the deployment
tradeoff at batch size 32 by combining latency, throughput, memory, and
forecasting accuracy. Batch size 32 is useful for comparing
high-throughput inference scenarios.

::: table*
[]{#tab:combined_bs32_results label="tab:combined_bs32_results"}
:::

## Latency and Throughput Analysis

Tiny Time Mixer achieved the lowest inference latency and highest
throughput across most configurations. At FP16 batch size 32, it reached
over 192,000 samples per second while maintaining a median latency of
only 0.177 ms. This indicates that compact time-series architectures can
be highly effective for high-throughput financial inference workloads.

LSTM and Transformer showed similar latency profiles, with median
latencies ranging from 0.15-0.25ms across batch sizes. Informer showed
significantly higher latency, particularly at larger batch sizes with
FP16 precision (3.53ms at batch size 32), likely due to its attention
mechanism complexity.

## Accuracy Analysis

The Informer model achieved the lowest forecasting error among the
evaluated architectures, with an MAE of approximately 0.107, RMSE of
approximately 0.593, and MAPE of approximately 32.8%. This indicates
that, under the current training configuration, the efficient
attention-based architecture provides the strongest point-forecasting
accuracy.

LSTM, Transformer, and Tiny Time Mixer achieved similar regression
errors in this benchmark, with MAE values around 0.257 and RMSE values
around 0.982. This may reflect the limited training schedule or the
specific characteristics of the financial dataset.

Directional Accuracy remained close to random-walk levels across models,
ranging from approximately 0.495 to 0.497. Informer FP32 obtained the
highest DA value at approximately 0.4967, followed by the other models
at approximately 0.4954. These results suggest that while some models
achieve low regression error, directional movement prediction remains
challenging.

## Precision Analysis

Mixed precision inference produced significant throughput benefits for
LSTM, Transformer, and Tiny Time Mixer. FP16 provided 2-3x throughput
improvements over FP32 at larger batch sizes, with minimal impact on
prediction accuracy. For Informer, FP16 reduced memory usage by
approximately 50% but did not improve throughput, likely due to the
computational overhead of its attention mechanism.

## Figures

The generated figures visualize latency, throughput-memory tradeoffs,
and forecasting accuracy. These figures should be uploaded to Overleaf
under a folder named `figures`.

<figure id="fig:latency_fp32" data-latex-placement="htbp">
<img src="./figures/latency_median_fp32.png" />
<figcaption>Median inference latency by model and batch size under FP32
precision.</figcaption>
</figure>

<figure id="fig:latency_fp16" data-latex-placement="htbp">
<img src="./figures/latency_median_fp16.png" />
<figcaption>Median inference latency by model and batch size under FP16
precision.</figcaption>
</figure>

<figure id="fig:throughput_memory_pareto" data-latex-placement="htbp">
<img src="./figures/throughput_memory_pareto_bs32.png" />
<figcaption>Throughput-memory tradeoff at batch size 32. Tiny Time Mixer
provides the strongest throughput-memory efficiency in the current
benchmark.</figcaption>
</figure>

<figure id="fig:accuracy_metrics" data-latex-placement="htbp">
<img src="./figures/accuracy_metrics.png" />
<figcaption>Forecasting accuracy metrics by model and precision. The
metrics include MAE, RMSE, MAPE, and directional accuracy.</figcaption>
</figure>

# Discussion

The results demonstrate that low-latency financial forecasting inference
is feasible on consumer-grade GPU hardware. Across the evaluated
configurations, Tiny Time Mixer provides the strongest inference
efficiency, achieving the highest throughput and lowest memory usage. At
batch size 32, Tiny Time Mixer reaches over 192,000 samples per second
while using approximately 10-11 MB of peak allocated GPU memory.

However, the most efficient model is not the most accurate model.
Informer achieves the lowest MAE, RMSE, and MAPE in the current
experiments, indicating that the efficient attention-based architecture
provides the strongest point-forecasting accuracy under the current
training configuration. This creates a clear tradeoff between deployment
efficiency and forecasting accuracy. Tiny Time Mixer is attractive for
high-throughput inference, while Informer is preferable when regression
accuracy is the primary objective.

LSTM and Transformer models provide competitive inference efficiency but
do not dominate either the accuracy or efficiency dimensions in the
current configuration. Their performance may improve with longer
training schedules, larger datasets, or more careful hyperparameter
tuning.

The strict no-fallback design improves the reliability of these
comparisons by ensuring that benchmark results correspond exactly to the
requested CUDA, backend, precision, and batch-size settings. This is
especially important when comparing FP32 and FP16 inference, since
silent fallback could otherwise produce misleading results.

# Limitations

The current work has several limitations:

- The experiments use a limited set of 25 financial instruments and
  should be expanded to include additional equities, indices, ETFs,
  commodities, fixed-income instruments, and foreign exchange data.

- Training uses 30 epochs, which may be insufficient for optimal
  convergence. Longer training schedules and broader hyperparameter
  search may significantly affect forecasting accuracy.

- Forecasting metrics are computed on supervised prediction targets and
  should not be interpreted as trading profitability without additional
  backtesting, transaction-cost modeling, and risk analysis.

- Directional accuracy remains close to random-walk levels, suggesting
  that additional covariates, richer features, or task-specific
  objectives may be necessary for stronger directional prediction.

- The current reported results focus on the PyTorch CUDA backend. ONNX
  Runtime, TensorRT, and other deployment backends remain important
  extensions.

- The benchmark is conducted on a single consumer GPU platform.
  Additional hardware comparisons are required to evaluate portability
  across devices.

- The current model set does not yet include heavier forecasting
  architectures such as Temporal Fusion Transformer.

# Future Work

Future work will extend the benchmark in several directions:

- Add Temporal Fusion Transformer as a stronger and more expressive
  forecasting benchmark.

- Evaluate ONNX Runtime and TensorRT inference backends for
  deployment-oriented acceleration.

- Compare additional consumer and data-center GPUs.

- Include more financial instruments, asset classes, and multi-asset
  forecasting setups.

- Evaluate longer training schedules and systematic hyperparameter
  tuning.

- Add confidence intervals and statistical significance analysis for
  latency, throughput, memory, and accuracy metrics.

- Study quantization, pruning, and other model-compression methods.

- Add backtesting metrics to connect forecast quality with portfolio or
  trading performance.

- Evaluate richer feature sets including technical indicators,
  macroeconomic variables, realized volatility, and cross-asset
  covariates.

## Temporal Fusion Transformer Extension

Temporal Fusion Transformer is a natural next benchmark for this study
because it represents a heavier and more expressive forecasting
architecture than the compact models currently evaluated. Unlike simple
recurrent or lightweight mixer models, TFT combines recurrent sequence
processing, gated residual networks, variable selection, static
covariate encoders, and interpretable temporal attention. This makes it
a stronger test of GPU memory pressure, inference latency, and
deployment feasibility.

In future experiments, TFT will be added to the benchmark matrix using
the same strict no-fallback protocol. The expected comparison will
evaluate whether the richer modeling capacity of TFT justifies its
additional inference cost relative to LSTM, Transformer, Informer, and
Tiny Time Mixer baselines. Since no TFT runs are included in the current
result table, TFT is reported as a planned extension rather than a
completed benchmark result.

# Conclusion

This paper presents a reproducible GPU benchmarking framework for
financial time-series forecasting inference. The framework evaluates
forecasting models under strict hardware, backend, precision, and
batch-size configurations. Completed PyTorch CUDA experiments on an
NVIDIA GeForce RTX 4070 Laptop GPU compare LSTM, Transformer, Tiny Time
Mixer, and Informer models under FP32 and FP16 precision across batch
sizes 1, 8, and 32.

The results show a clear tradeoff between inference efficiency and
forecasting accuracy. Tiny Time Mixer provides the strongest deployment
efficiency, achieving the highest throughput (up to 192K samples/sec)
and lowest GPU memory usage. Informer provides the strongest regression
accuracy, achieving the lowest MAE (0.107), RMSE (0.593), and MAPE
(32.8%) under the current training configuration. LSTM and Transformer
models provide competitive inference efficiency but do not dominate the
current benchmark results.

Overall, the benchmark demonstrates that consumer GPU hardware can
support low-latency financial forecasting inference while preserving
strict experimental reproducibility. Future work will extend the
benchmark to Temporal Fusion Transformer, ONNX Runtime, TensorRT,
additional hardware platforms, richer datasets, and backtesting-oriented
financial evaluation.

# Acknowledgment {#acknowledgment .unnumbered}

The author thanks the open-source Python, PyTorch, CUDA, Yahoo Finance,
NumPy, pandas, matplotlib, and seaborn communities for enabling
reproducible financial machine learning experimentation.

::: thebibliography
00

S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural
Computation*, vol. 9, no. 8, pp. 1735--1780, 1997.

A. Vaswani et al., "Attention Is All You Need," in *Advances in Neural
Information Processing Systems*, 2017.

H. Zhou et al., "Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting," in *Proceedings of the AAAI
Conference on Artificial Intelligence*, 2021.

A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep
Learning Library," in *Advances in Neural Information Processing
Systems*, 2019.

NVIDIA, "CUDA Toolkit Documentation," Available:
<https://docs.nvidia.com/cuda/>

R. Aroussi, "yfinance: Yahoo Finance Market Data Downloader," Available:
<https://github.com/ranaroussi/yfinance>

ONNX Runtime Developers, "ONNX Runtime," Available:
<https://onnxruntime.ai/>

NVIDIA, "TensorRT Documentation," Available:
<https://docs.nvidia.com/deeplearning/tensorrt/>
:::
