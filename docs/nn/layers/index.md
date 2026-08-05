# Neural network layers

Every `ion.nn` layer is an immutable JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html). Constructors use a keyword-only `key` for parameter initialization, and layers work directly with native JAX transformations.

## Choose a layer

| Family | Layers |
|---|---|
| [Linear](linear.md) | `Linear` |
| [Convolution](conv.md) | `Conv`, `ConvTranspose` |
| [Attention](attention.md) | `MultiHeadAttention` |
| [Normalization](norm.md) | `BatchNorm`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `SpectralNorm` |
| [Recurrent](recurrent.md) | `RNN`, `LSTM`, `GRU` and their cells |
| [State Space](ssm.md) | `S4D`, `S5`, `LRU` and their cells |
| [Embedding](embedding.md) | `Embedding` |
| [Positional](positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `sinusoidal`, `alibi` |
| [Pooling](pool.md) | `MaxPool`, `AvgPool` |
| [Dropout](dropout.md) | `Dropout` |
| [LoRA](lora.md) | `LoRALinear` |
| [Identity](identity.md) | `Identity` |
| [MLP](mlp.md) | `MLP` |
| [Sequential](sequential.md) | `Sequential` |

`MLP` and `Sequential` assemble other layers but remain ordinary `Module` pytrees. The [NN guide](../guide.md#array-conventions) documents the shared array, batching, shape, and dtype conventions.
