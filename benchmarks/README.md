# Benchmarks

Ion is benchmarked against Equinox, Flax NNX, and PyTorch. PyTorch is measured
in both eager and compiled modes. The suite compares three model families at
three sizes:

| Model | Tiny | Small | Medium |
|---|---|---|---|
| MLP | 4 × 128 (0.29M) | 8 × 512 (2.61M) | 12 × 2048 (46.11M) |
| ResNet | 1/1/1/1, width 32 (1.49M) | 2/2/2/2, width 64 (11.69M) | 3/4/23/3, width 64 (41.87M) |
| GPT | 2L/128D/4H, seq 128 (4.49M) | 6L/384D/6H, seq 256 (22.91M) | 12L/768D/12H, seq 512 (109.55M) |

All models use float32 master parameters and bfloat16 computation. ResNets use
GroupNorm so the benchmark does not require mutable running statistics.

## Metrics

- **Forward** measures the model call without its loss.
- **Forward + backward** measures the loss and gradient calculation.
- **Full step** additionally applies an AdamW update.
- **First step** measures the first call to the compiled full training step.
- **Compile** is first-step latency minus median warmed-step latency.
- **Throughput** is derived from median full-step latency.
- **Peak memory** measures a full training step using the framework's native
  allocator statistics.

Compile is deliberately a user-facing estimate rather than a compiler-internal
measurement. Every framework, mode, model, size, and repetition runs in a fresh
process, preventing in-memory caches from leaking between comparable results.
Forward, forward + backward, and the full training step are compiled once within
that process. First-step latency, compile time, throughput, and peak memory are
collected from the same full-step execution. Persistent compiler caches should
be disabled or pointed at a fresh directory when collecting publishable results.

JAX does not expose a way to reset its allocator's peak counter. Its peak memory
therefore includes initialization, compilation, warm-up, and the measured steps.
PyTorch's counter is reset immediately before the measured steps. This limitation
must be stated alongside published memory results.

## Fairness

- Inputs and targets are generated before timing and remain on the device.
- Every timed result is synchronized with the accelerator.
- Architectures, losses, optimizer, batch sizes, and parameter counts are
  equivalent across frameworks.
- Ion and NNX use channels-last convolution layouts. Equinox and PyTorch use
  their native channels-first layouts.
- GPT uses causal scaled dot-product attention and tied token embeddings.
- Dropout, data loading, distributed execution, and gradient accumulation are
  outside the timed region.

Parameter initialization is framework-native. Exact outputs are not expected to
match, but shapes, parameter counts, and training semantics must match.

## Running

Install the benchmark dependencies:

```bash
uv sync --group benchmarks
```

Run one short case:

```bash
uv run --group benchmarks python -m benchmarks.runner \
  ion mlp tiny full_step --warmup 2 --steps 10
```

Run the complete matrix:

```bash
uv run --group benchmarks python -m benchmarks.run_all \
  --output benchmarks/results/reference-gpu
```

The full matrix contains 135 isolated process runs with the default three
repetitions, five warm-up steps, and 50 measured steps. Completed cases are
skipped automatically, so an interrupted suite can be resumed with the same
command. Pass `--overwrite` to replace existing results.

Use filters while developing:

```bash
uv run --group benchmarks python -m benchmarks.run_all \
  --frameworks ion nnx --models mlp --sizes tiny --repetitions 1 \
  --warmup 2 --steps 10 --output benchmarks/results/smoke
```

For a more exhaustive publication run:

```bash
uv run --group benchmarks python -m benchmarks.run_all \
  --repetitions 5 --warmup 10 --steps 100 \
  --output benchmarks/results/publication
```

Generate a Markdown summary:

```bash
uv run --group benchmarks python -m benchmarks.report \
  benchmarks/results/reference-gpu --output benchmarks/results/report.md
```

Render interactive HTML plots:

```bash
uv run --group benchmarks python -m benchmarks.plot \
  benchmarks/results/reference-gpu
```

This writes latency, throughput, compile-time, first-step, and peak-memory
figures to `benchmarks/results/reference-gpu/plots`. Pass `--inline` to create
fully self-contained HTML files instead of loading Plotly from its CDN.
Reports and plots automatically use the largest common number of repetitions
and measured samples for each comparison.

For publishable results, record the GPU, driver, CUDA, Python, JAX, XLA, Flax,
Equinox, and PyTorch versions. Use an otherwise idle machine with fixed GPU
power and clock settings where possible.
