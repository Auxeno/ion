# Layers

Conventions and design decisions across Ion's layer library.

## Input Format

All layers use **channels-last** ordering. 

| Domain | Format | Example |
|--------|--------|---------|
| Vector data | `(..., features)` | `(batch, 256)` |
| 1D (sequences) | `(..., length, channels)` | `(batch, 128, 64)` |
| 2D (images) | `(..., height, width, channels)` | `(batch, 32, 32, 3)` |
| Attention | `(..., seq, dim)` | `(batch, 128, 512)` |
| Recurrent | `(..., time, features)` | `(batch, 50, 64)` |

Channels-last is the most typical format for image data and is followed by Flax and TensorFlow. PyTorch and Equinox use the channels-first convention.

### Flexible Batch Dimensions

All layers accept arbitrary leading batch dimensions — zero, one, or many. The same layer works on a single example, a batch, or nested batches with no code changes.

```python
linear = nn.Linear(4, 8, key=key)

linear(x)          # (4,)       ->  (8,)       no batch
linear(x_batched)  # (3, 4)     ->  (3, 8)     one batch dim
linear(x_multi)    # (5, 3, 4)  ->  (5, 3, 8)  multiple batch dims
```

This works the same way for spatial (and all other) layers:

```python
conv = nn.Conv(2, 3, 16, kernel_size=3, padding=1, key=key)

conv(x)          # (32, 32, 3)        ->  (32, 32, 16)        no batch
conv(x_batched)  # (8, 32, 32, 3)     ->  (8, 32, 32, 16)     one batch dim
conv(x_multi)    # (2, 8, 32, 32, 3)  ->  (2, 8, 32, 32, 16)  multiple batch dims
```

This is a deliberate design choice favouring ease of use. PyTorch is inconsistent here as some layers (e.g. `nn.Linear`) support arbitrary batch dims while others (e.g. `nn.Conv2d`) only accept unbatched or single-batch input. Equinox requires `jax.vmap` over a model call to add a batch dimension, which is the most explicit approach and best for debugging. Flax and Ion let the user call any layer with or without batch dims freely.

The trade-off for the Flax/Ion approach is users won't catch an accidentally wrong number of dimensions since an extra or missing dim will silently produce wrong-shaped outputs rather than an error, so manage your array shapes carefully.

## Shape Annotations

Single-letter dimension labels are used in `jaxtyping` annotations and einsum strings. These follow conventions from the JAX ecosystem.

The same letter can mean different things in different layers — meaning is determined by context, not globally.

### General

| Label | Meaning | Used in |
|-------|---------|---------|
| `d` | model / feature dimension | linear, attention, norm, embedding, positional |
| `i` | input features | linear, recurrent, lora |
| `o` | output features | linear, lora |
| `r` | rank | lora |
| `v` | vocabulary size | embedding |
| `...` | arbitrary batch dimensions | everywhere |

### Attention

| Label | Meaning | Notes |
|-------|---------|-------|
| `d` | model dimension | total embedding size |
| `h` | number of heads | matches Haiku / original paper convention |
| `k` | per-head dimension (`d_k`) | from "Attention Is All You Need" |
| `s` | query (source) sequence position | |
| `t` | key/value (target) sequence position | distinct from `s` in cross-attention |
| `i` | QKV index | literal 3 (self-attn) or 2 (cross-attn KV) |

Einsum patterns:

```
QKV projection:     ...d, dihk -> ...ihk
Attention logits:   ...shk, ...thk -> ...hst
Attention output:   ...hst, ...thk -> ...shk
Output projection:  ...hk, hkd -> ...d
```

### Recurrent

| Label | Meaning | Notes |
|-------|---------|-------|
| `i` | input features | |
| `h` | hidden dimension | same letter as attention heads — context resolves it |
| `g` | gate dimension | `4h` for LSTM, `3h` for GRU |
| `t` | time steps | sequence dimension |

### Convolution & Spatial

| Label | Meaning | Used in |
|-------|---------|---------|
| `c` | channels | conv, pool, upsample |
| `h` | height | pool, upsample, conv (2D) |
| `w` | width | pool, upsample, conv (2D) |
| `l` | length | pool (1D), upsample (1D) |

### Positional Encodings

| Label | Meaning | Used in |
|-------|---------|---------|
| `s` | sequence position | sinusoidal, learned, rope |
| `d` | feature dimension | sinusoidal, learned, rope |
| `h` | number of heads | alibi |

## Spatial Layers

Convolution, pooling, and upsampling layers are N-dimensional — the first argument `num_spatial_dims` controls dimensionality. This keeps the API surface small while supporting 1D, 2D, 3D, and beyond with the same class.

```python
Conv(1, 3, 16, kernel_size=5, key=key)           # Conv1d
Conv(2, 3, 16, kernel_size=3, key=key)           # Conv2d
ConvTranspose(2, 16, 3, kernel_size=3, key=key)  # ConvTranspose2d
MaxPool(2, kernel_size=2)                        # MaxPool2d
AvgPool(1, kernel_size=3, padding=1)             # AvgPool1d
Upsample(2, scale_factor=2)                      # Upsample2d
```

Scalar values for `kernel_size`, `stride`, `padding`, etc. are broadcast across all spatial dimensions. Tuples give per-dimension control.

## Weight Initialization

Each layer family uses init schemes suited to its typical activation:

| Layer | Weights | Bias |
|-------|---------|------|
| Linear, Conv, MLP | He normal | zeros |
| Attention, Embedding, Positional | Truncated normal (std=0.02) | zeros |
| Recurrent (input) | Glorot uniform | zeros (LSTM forget gate: ones) |
| Recurrent (hidden) | Orthogonal | — |
| Norm | scale=1, bias=0 | — |

**Linear, Conv, and MLP** default to He normal, which assumes ReLU activation. If using a different activation (tanh, GELU, sigmoid, etc.), pass a different `w_init` — for example `jax.nn.initializers.glorot_uniform()` for tanh/sigmoid or `jax.nn.initializers.lecun_normal()` for SELU. Using He normal with non-ReLU activations can cause vanishing signals.

**Attention and Embedding** use truncated normal with a small std (0.02), standard practice from GPT-2 and BERT. This is activation-agnostic since attention weights are followed by softmax, not a pointwise activation.

**Recurrent** layers use Glorot uniform for input-to-hidden weights and orthogonal for hidden-to-hidden weights. Orthogonal init preserves gradient norms across time steps, reducing vanishing/exploding gradients in long sequences. LSTM forget gate bias is initialized to 1 to encourage remembering early in training.

## Stateless Design

All layers are stateless and frozen after `__init__`. Two patterns handle state:

- **Dropout** takes a `key` argument at call time for stochastic masking.
- **BatchNorm** takes a `state` argument (running mean/var) and returns updated state alongside the output. The caller manages state explicitly.
