# NN

`ion.nn` contains neural network layers built on Ion's
[`Module`](../core/module.md). Every layer is an immutable JAX pytree and works
directly with `jax.jit`, `jax.grad`, and `jax.vmap`.

This page starts from an input array, applies a layer, builds a small model, and
then collects the input conventions shared by the layer library. For data with
explicit connections between items, see the [GNN tutorial](../gnn/index.md).

## Start with Features

The final axis of an input array stores features. Here there are four items and
three features per item:

```python
import jax
import jax.numpy as jnp

from ion import nn

x = jnp.array([
    [1.0, 0.2, -0.4],
    [0.3, 0.8, 0.1],
    [-0.2, 0.5, 0.9],
    [0.7, -0.1, 0.4],
])

x.shape  # (4, 3)
```

The leading axis can be a batch, a sequence, or another collection of items.
A `Linear` layer transforms the final feature dimension and preserves every
leading dimension:

```python
key = jax.random.key(0)
linear = nn.Linear(
    in_dim=3,
    out_dim=8,
    key=key,
)

y = linear(x)
y.shape  # (4, 8)
```

Each row is transformed independently by the same weights. The layer does not
know whether two rows are related. Graph layers differ by receiving
`senders` and `receivers` arrays that define those relationships.

## Build a Model

Composite modules are constructed and called like individual layers. An MLP
lists the feature dimension at every stage:

```python
model = nn.MLP(
    dims=[3, 16, 16, 2],
    key=jax.random.key(1),
)

logits = model(x)
logits.shape  # (4, 2)
```

The same pattern applies when defining a custom module:

```python
class Classifier(nn.Module):
    hidden: nn.Linear
    output: nn.Linear

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key):
        key_hidden, key_output = jax.random.split(key)
        self.hidden = nn.Linear(in_dim, hidden_dim, key=key_hidden)
        self.output = nn.Linear(hidden_dim, num_classes, key=key_output)

    def __call__(self, x):
        x = jax.nn.relu(self.hidden(x))
        return self.output(x)
```

Since the model is a pytree, native JAX transforms take the model directly:

```python
jitted_model = jax.jit(model)

def loss_fn(model, x, targets):
    predictions = model(x)
    return jnp.mean((predictions - targets) ** 2)

targets = jnp.zeros((4, 2))
grads = jax.grad(loss_fn)(model, x, targets)
```

There is no `ion.jit` or `ion.grad`.

## Layer Reference

| Family | Layers |
|---|---|
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

`MLP` and `Sequential` assemble other layers but remain ordinary `Module`
pytrees.

### Input Formats

Layers use channels-last ordering:

| Domain | Format | Example |
|---|---|---|
| Vector data | `(batch, features)` | `(32, 256)` |
| Sequences | `(batch, length, channels)` | `(32, 128, 64)` |
| Images | `(batch, height, width, channels)` | `(32, 32, 32, 3)` |
| Attention | `(batch, sequence, dimension)` | `(32, 128, 512)` |
| Recurrent | `(batch, time, features)` | `(32, 50, 64)` |

Pointwise layers such as `Linear` preserve arbitrary leading dimensions.
Structural layers document any stricter rank or batching requirements on their
reference pages.

### Shape Labels

Single-letter dimension labels appear in type annotations and einsum strings.
Their meaning is local to each layer:

| Label | Common meaning |
|---|---|
| `...` | Arbitrary leading dimensions |
| `b` | Batch |
| `d` | Model or feature dimension |
| `i`, `o` | Input and output features |
| `r` | LoRA rank |
| `v` | Vocabulary size |

The same letter can have a different meaning in a different layer. The
signature and example on each reference page define its local shapes.
