# NN

`ion.nn` is the neural network layer library built on the [core](../core/module.md). Every layer is a `Module`: an immutable pytree that works directly with `jax.jit`, `jax.grad`, and `jax.vmap`. Layers are constructed with their dimensions and a keyword-only `key`, then called on a batched input.

The layer families are below, followed by the input and shape-label conventions
shared across them.

## Layers

| Family | Layers |
|--------|--------|
| [Linear](layers/linear.md) | `Linear`, `Identity` |
| [Convolution](layers/conv.md) | `Conv`, `ConvTranspose` |
| [Attention](layers/attention.md) | `SelfAttention`, `CrossAttention` |
| [Normalization](layers/norm.md) | `LayerNorm`, `RMSNorm`, `GroupNorm` |
| [Recurrent](layers/recurrent.md) | `RNN`, `LSTM`, `GRU` and their cells |
| [State Space](layers/ssm.md) | `S4D`, `S5`, `LRU` and their cells |
| [Embedding](layers/embedding.md) | `Embedding` |
| [Positional](layers/positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `sinusoidal`, `alibi` |
| [Pooling](layers/pool.md) | `MaxPool`, `AvgPool` |
| [Dropout](layers/dropout.md) | `Dropout` |
| [LoRA](layers/lora.md) | `LoRALinear` |
| [MLP](layers/mlp.md) | `MLP` |
| [Sequential](layers/sequential.md) | `Sequential` |

`MLP` and `Sequential` are composite modules that assemble other layers, but they are constructed and called like any other layer.

## Input Format

All layers use **channels-last** ordering.

| Domain | Format | Example |
|--------|--------|---------|
| Vector data | `(batch, features)` | `(32, 256)` |
| 1D (sequences) | `(batch, length, channels)` | `(32, 128, 64)` |
| 2D (images) | `(batch, height, width, channels)` | `(32, 32, 32, 3)` |
| Attention | `(batch, seq, dim)` | `(32, 128, 512)` |
| Recurrent | `(batch, time, features)` | `(32, 50, 64)` |

Channels-last is the most typical format for image data and is followed by Flax and TensorFlow. PyTorch and Equinox use the channels-first convention.

## Shape Annotations

Single-letter dimension labels are used in `jaxtyping` annotations and einsum strings. These follow conventions from the JAX ecosystem.

The same letter can mean different things in different layers. Meaning is determined by context, not globally.

### General

| Label | Meaning | Used in |
|-------|---------|---------|
| `d` | model / feature dimension | linear, attention, norm, embedding, positional |
| `i` | input features | linear, recurrent, lora |
| `o` | output features | linear, lora |
| `r` | rank | lora |
| `v` | vocabulary size | embedding |
| `b` | batch dimension | everywhere |
| `...` | arbitrary batch dimensions	 | everywhere |
