# Benchmarks

Ion is benchmarked against Equinox, Flax NNX, and PyTorch on three representative model families. PyTorch is shown in both eager and compiled modes and JAX frameworks are compiled with `jax.jit`.

## Setup

| | |
|---|---|
| **Accelerator** | NVIDIA H100 80GB HBM3 |
| **Precision** | float32 master parameters, bfloat16 computation |
| **Ion** | 0.11.1 |
| **JAX stack** | JAX 0.11.0, jaxlib 0.11.0 |
| **Other frameworks** | Equinox 0.13.8, Flax 0.12.8, PyTorch 2.13.0+cu130 |
| **Runtime** | Python 3.12.3, CUDA 13.0, cuDNN 9.2 |

## Workloads

| Model | Tiny | Small | Medium |
|---|---|---|---|
| **MLP** | 4 × 128 (0.29M) | 8 × 512 (2.61M) | 12 × 2048 (46.11M) |
| **ResNet** | 1/1/1/1, width 32 (1.49M) | 2/2/2/2, width 64 (11.69M) | 3/4/23/3, width 64 (41.87M) |
| **GPT** | 2L/128D/4H, seq 128 (4.49M) | 6L/384D/6H, seq 256 (22.91M) | 12L/768D/12H, seq 512 (109.55M) |

## Training throughput

Samples per second for MLP and ResNet; tokens per second for GPT. Higher is better.

<iframe
  class="benchmark-plot"
  src="../assets/benchmarks/throughput.html"
  title="Training throughput benchmark results"
  loading="eager"
></iframe>

!!! note
    PyTorch's compiled GPT uses FlashAttention by default, while the JAX implementations benchmarked here do not, although they can be configured to use it. This gives PyTorch an advantage in GPT throughput and memory usage.

## Execution time

Warmed forward, forward-and-backward, and complete AdamW training-step latency. Lower is better.

<iframe
  class="benchmark-plot benchmark-plot--latency"
  src="../assets/benchmarks/latency.html"
  title="Execution time benchmark results"
  loading="lazy"
></iframe>

## Compile time

Estimated as first-step latency minus warmed full-step latency. Lower is better.

<iframe
  class="benchmark-plot"
  src="../assets/benchmarks/compile-time.html"
  title="Compile time benchmark results"
  loading="lazy"
></iframe>

## First training step

End-to-end latency of the first compiled training step. Lower is better.

<iframe
  class="benchmark-plot"
  src="../assets/benchmarks/first-step.html"
  title="First training step benchmark results"
  loading="lazy"
></iframe>

## Peak device memory

Peak accelerator memory observed during a complete training step. Lower is better.

<iframe
  class="benchmark-plot"
  src="../assets/benchmarks/peak-memory.html"
  title="Peak device memory benchmark results"
  loading="lazy"
></iframe>

The [benchmark source](https://github.com/auxeno/ion/tree/51cba60/benchmarks) contains the model implementations and complete measurement procedure.
