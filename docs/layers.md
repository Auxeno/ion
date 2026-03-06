# Layers

Conventions and design decisions across Ion's layer library.

## Data Format

All layers use **channels-last** ordering to match JAX and image data conventions:

| Domain | Format | Example |
|--------|--------|---------|
| 1D (sequences) | `(..., length, channels)` | `(batch, 128, 64)` |
| 2D (images) | `(..., height, width, channels)` | `(batch, 32, 32, 3)` |
| Attention | `(..., seq, dim)` | `(batch, 128, 512)` |
| Recurrent | `(..., time, features)` | `(batch, 50, 64)` |

Batch dimensions are always leading and handled via `...` (ellipsis) in type hints, so all layers support arbitrary batch shapes without explicit batch logic.

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
QKV projection:       ...d, dihk -> ...ihk
Attention logits:     ...shk, ...thk -> ...hst
Attention output:     ...hst, ...thk -> ...shk
Output projection:    ...hk, hkd -> ...d
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
