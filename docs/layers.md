# Layers

Conventions and design decisions across Ion's layer library.

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

All layers expect at least one leading batch dimension. Structural layers (Conv, Pool, LSTM, GRU, GroupNorm) require exactly the right number of dimensions and will error on incorrect rank. Pointwise layers (Linear, LayerNorm, Embedding, etc.) operate on the last dimension and naturally handle any number of leading dims.

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

### Attention

| Label | Meaning | Notes |
|-------|---------|-------|
| `d` | model dimension | total embedding size |
| `h` | number of heads | matches original paper convention |
| `k` | per-head dimension (`d_k`) | from "Attention Is All You Need" |
| `s` | query (source) sequence position | |
| `t` | key/value (target) sequence position | distinct from `s` in cross-attention |
| `i` | QKV index | literal 3 (self-attn) or 2 (cross-attn KV) |

Einsum patterns:

```
QKV projection:     bd, dihk -> bihk
Attention logits:   bshk, bthk -> bhst
Attention output:   bhst, bthk -> bshk
Output projection:  bhk, hkd -> bd
```

### Recurrent

| Label | Meaning | Notes |
|-------|---------|-------|
| `i` | input features | |
| `h` | hidden dimension | same letter as attention heads; context resolves it |
| `g` | gate dimension | `4h` for LSTM, `3h` for GRU |
| `t` | time steps | sequence dimension |

### Convolution & Spatial

| Label | Meaning | Used in |
|-------|---------|---------|
| `c` | channels | conv, pool |
| `h` | height | pool, conv (2D) |
| `w` | width | pool, conv (2D) |
| `l` | length | pool (1D) |

### Positional Encodings

| Label | Meaning | Used in |
|-------|---------|---------|
| `s` | sequence position | sinusoidal, learned, rope |
| `d` | feature dimension | sinusoidal, learned, rope |
| `h` | number of heads | alibi |

## Spatial Layers

Convolution and pooling layers are N-dimensional. This keeps the API surface small while supporting 1D, 2D, 3D, and beyond with the same class.

All spatial layers infer the spatial rank from `kernel_shape`, which must be a tuple.

```python
Conv(3, 16, kernel_shape=(5,), key=key)             # Conv1d
Conv(3, 16, kernel_shape=(3, 3), key=key)           # Conv2d
ConvTranspose(16, 3, kernel_shape=(3, 3), key=key)  # ConvTranspose2d
MaxPool(kernel_shape=(2, 2))                        # MaxPool2d
AvgPool(kernel_shape=(3,), padding=1)               # AvgPool1d
```

Scalar values for `stride`, `padding`, `dilation`, etc. are broadcast across all spatial dimensions. Tuples give per-dimension control.

## Weight Initialization

Each layer family uses init schemes suited to its typical activation:

| Layer | Weights | Bias |
|-------|---------|------|
| Linear, Conv, MLP | He normal | zeros |
| Attention, Embedding, Positional | Truncated normal (std=0.02) | zeros |
| Recurrent (input) | Glorot uniform | zeros (LSTM forget gate: ones) |
| Recurrent (hidden) | Orthogonal | - |
| Norm | scale=1, bias=0 | - |

**Linear, Conv, and MLP** default to He normal, which assumes ReLU activation. If using a different activation (tanh, GELU, sigmoid, etc.), pass a different `w_init` like `jax.nn.initializers.glorot_uniform()` for tanh/sigmoid or `jax.nn.initializers.lecun_normal()` for SELU. Using He normal with non-ReLU activations can cause vanishing signals.

**Attention and Embedding** use truncated normal with a small std (0.02), standard practice from GPT-2 and BERT. This is activation-agnostic since attention weights are followed by softmax, not a pointwise activation.

**Recurrent** layers use Glorot uniform for input-to-hidden weights and orthogonal for hidden-to-hidden weights. Orthogonal init preserves gradient norms across time steps, reducing vanishing/exploding gradients in long sequences. LSTM forget gate bias is initialized to 1 to encourage remembering early in training.

## Stateless Design

All layers are stateless and frozen after `__init__`. Two patterns handle state:

- **Dropout** takes a `key` argument at call time for stochastic masking.
- **BatchNorm** stores running statistics on the module. A `training` flag controls whether batch or running stats are used. The `update` method returns a new module with updated running statistics.
