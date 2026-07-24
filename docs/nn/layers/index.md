# Neural network layers

Every `ion.nn` layer is an immutable JAX pytree. Constructors use a keyword-only
`key` for parameter initialization, and layers work directly with native JAX
transformations.

## Choose a layer

| Family | Layers |
|---|---|
| [Linear](linear.md) | `Linear`, `Identity` |
| [Convolution](conv.md) | `Conv`, `ConvTranspose` |
| [Attention](attention.md) | `SelfAttention`, `CrossAttention` |
| [Normalization](norm.md) | `LayerNorm`, `RMSNorm`, `GroupNorm` |
| [Recurrent](recurrent.md) | `RNN`, `LSTM`, `GRU` and their cells |
| [State Space](ssm.md) | `S4D`, `S5`, `LRU` and their cells |
| [Embedding](embedding.md) | `Embedding` |
| [Positional](positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `sinusoidal`, `alibi` |
| [Pooling](pool.md) | `MaxPool`, `AvgPool` |
| [Dropout](dropout.md) | `Dropout` |
| [LoRA](lora.md) | `LoRALinear` |
| [MLP](mlp.md) | `MLP` |
| [Sequential](sequential.md) | `Sequential` |

`MLP` and `Sequential` assemble other layers but remain ordinary `Module`
pytrees. The [NN guide](../guide.md#array-conventions) documents the shared
array, batching, shape, and dtype conventions.
