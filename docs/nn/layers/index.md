# Neural network layers

Every `ion.nn` layer is an immutable JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html). Constructors use a keyword-only `key` for parameter initialization, and layers work directly with native JAX transformations.

## Choose a layer

| Family | Layers |
|---|---|
| [Linear](linear.md) | `Linear`, `Identity` |
| [Convolution](conv.md) | `Conv`, `ConvTranspose` |
| [Attention](attention.md) | `MultiHeadAttention` |
| [Normalization](norm.md) | `LayerNorm`, `RMSNorm`, `BatchNorm`, `GroupNorm`, `SpectralNorm` |
| [Recurrent](recurrent.md) | `RNN`, `LSTM`, `GRU` and their cells |
| [State Space](ssm.md) | `S4D`, `S5` and their cells |
| [Embedding](embedding.md) | `Embedding` |
| [Positional](positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `SinusoidalPositionalEmbedding` |
| [Pooling](pool.md) | `MaxPool`, `AvgPool` |
| [Stochastic](stochastic.md) | `Dropout`, `DropPath` |
| [Blocks](blocks.md) | `MLP`, `Sequential`, `Residual`, `Bidirectional`, `Ensemble` |

Blocks assemble other layers but remain ordinary `Module` pytrees. The [NN guide](../guide.md#array-conventions) documents the shared array, batching, shape, and dtype conventions.
