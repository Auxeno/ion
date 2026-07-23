# NN

`ion.nn` is the neural network layer library built on the [core](../core/module.md). Every layer is a `Module`: an immutable pytree that works directly with `jax.jit`, `jax.grad`, and `jax.vmap`. Layers are constructed with their dimensions and a keyword-only `key`, then called on a batched input.

```python
import jax
import ion.nn as nn

key = jax.random.key(0)
linear = nn.Linear(4, 8, key=key)
y = linear(jax.numpy.ones((32, 4)))  # (32, 4) -> (32, 8)
```

The layer families are below, followed by the input, batching, shape-label, and
initializer conventions shared across them.

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

### Batch Dimensions

All layers expect at least one leading batch dimension. Structural layers (Conv, Pool, RNN, LSTM, GRU) require exactly the right number of dimensions and will error on incorrect rank. Pointwise layers (Linear, LayerNorm, Embedding, etc.) operate on the last dimension and naturally handle any number of leading dims; GroupNorm likewise operates on the trailing spatial and channel dimensions.

```python
linear = nn.Linear(4, 8, key=key)
x = jnp.ones((32, 4))
linear(x)  # (32, 4) -> (32, 8)

conv = nn.Conv(3, 16, kernel_shape=(3, 3), padding=1, key=key)
x = jnp.ones((32, 28, 28, 3))
conv(x)  # (32, 28, 28, 3) -> (32, 28, 28, 16)
```

Use `jax.vmap` for inputs with an multiple batch dimensions:

```python
x = jnp.ones((4, 32, 28, 28, 3))
jax.vmap(conv)(x)  # (4, 32, 28, 28, 3) -> (4, 32, 28, 28, 16)

x = jnp.ones((2, 4, 32, 28, 28, 3))
jax.vmap(jax.vmap(conv))(x)  # (2, 4, 32, 28, 28, 3) -> (2, 4, 32, 28, 28, 16)
```

This design catches shape errors. Passing the wrong number of dimensions to a Conv or LSTM will raise an error rather than silently reshaping. 

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

## Weight Initialization

Every `w_init` / `b_init` argument takes an `Initializer` (`jax.nn.initializers.Initializer`): a callable `(key, shape, dtype) -> Array`. Pass any factory from `jax.nn.initializers`, such as `he_normal()`, `glorot_uniform()`, `truncated_normal(0.02)`, or `zeros`. In the API reference these type and default names are links into the JAX docs, and each layer's default is shown in its signature.

Layer-specific defaults and their rationale are documented with each
[layer family](#layers).
