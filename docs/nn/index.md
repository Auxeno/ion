# NN

`ion.nn` is the layer library built on the [core](../core/module.md). Every layer is a `Module`: an immutable pytree that works directly with `jax.jit`, `jax.grad`, and `jax.vmap`. Layers are constructed with their dimensions and a keyword-only `key`, then called on a batched input.

```python
import jax
import ion.nn as nn

key = jax.random.key(0)
linear = nn.Linear(4, 8, key=key)
y = linear(jax.numpy.ones((32, 4)))  # (32, 4) -> (32, 8)
```

Start with [Conventions](conventions.md) for the cross-cutting rules (input format, shape labels, batching, initialization, masking) that apply across all layers, then see each family below.

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

## Blocks

Composite modules that assemble layers into reusable architectures live in [Blocks](blocks.md): `MLP` and `Sequential`.
