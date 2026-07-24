# Overview

Ion is a minimal harness for building and training neural networks in JAX. It has a small core of three pieces, with NN and GNN layers built on top.

## The core

- [**Param**](core/param.md) wraps a JAX array and marks it trainable or frozen.
- [**Module**](core/module.md) is the base class for models and layers. A model is an immutable [pytree](https://docs.jax.dev/en/latest/pytrees.html): params and submodules at the leaves, everything else in the structure.
- [**Optimizer**](core/optimizer.md) wraps any [optax](https://github.com/google-deepmind/optax) transform and updates a model, automatically partitioning out non-trainable parameters.

The whole core is under a thousand lines of code, small enough to read in an afternoon.

## Building a model

Subclass `Module`, declare fields as class annotations, and assign them in `__init__`:

```python
import jax, typing
import ion.nn as nn

class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    activation: typing.Callable

    def __init__(self, activation=jax.nn.relu, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(4, 16, key=keys[0])
        self.layer_2 = nn.Linear(16, 3, key=keys[1])
        self.activation = activation

    def __call__(self, x):
        return self.layer_2(self.activation(self.layer_1(x)))

model = MLP(key=jax.random.key(0))
```

Fields can hold params, submodules, callables, and static configuration.

```python
>>> model
MLP(
  layer_1=Linear(
    w=Param(f32[4, 16], trainable=True),
    b=Param(f32[16], trainable=True),
  ),
  layer_2=Linear(
    w=Param(f32[16, 3], trainable=True),
    b=Param(f32[3], trainable=True),
  ),
  activation=relu,
)
```

In a terminal, a pretty printer gives the tree as text. In IPython/Jupyter environments, models render with [Treescope](https://github.com/google-deepmind/treescope).

## Neural network layers

Each [`ion.nn`](nn/layers/index.md) layer is a `Module`, constructed with a `key` for weight initialization:

| Family | Layers |
|--------|--------|
| [Linear](nn/layers/linear.md) | `Linear`, `Identity` |
| [Convolution](nn/layers/conv.md) | `Conv`, `ConvTranspose` |
| [Attention](nn/layers/attention.md) | `SelfAttention`, `CrossAttention` |
| [Normalization](nn/layers/norm.md) | `LayerNorm`, `RMSNorm`, `GroupNorm` |
| [Recurrent](nn/layers/recurrent.md) | `RNN`, `LSTM`, `GRU` |
| [State Space](nn/layers/ssm.md) | `S4D`, `S5`, `LRU` |
| [Embedding](nn/layers/embedding.md) | `Embedding` |
| [Positional](nn/layers/positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `sinusoidal`, `alibi` |
| [Pooling](nn/layers/pool.md) | `MaxPool`, `AvgPool` |
| [Dropout](nn/layers/dropout.md) | `Dropout` |
| [LoRA](nn/layers/lora.md) | `LoRALinear` |
| [Composite](nn/layers/mlp.md) | `MLP`, `Sequential` |

```python
import ion.nn as nn

attn = nn.SelfAttention(64, num_heads=8, key=jax.random.key(0))
y = attn(jax.numpy.ones((32, 16, 64)))
```

The [NN guide](nn/guide.md) builds and trains a model and collects the shared
array conventions. The [layer reference](nn/layers/index.md) lists the
available families.

## Graph neural network layers

[`ion.gnn`](gnn/guide.md) provides graph layers and message-passing operations.

| Family | Layers |
|--------|--------|
| [GCN](gnn/layers/gcn.md) | `GCNConv` |
| [GAT](gnn/layers/gat.md) | `GATConv`, `GATv2Conv` |
| [GIN](gnn/layers/gin.md) | `GINConv` |
| [Operations](gnn/operations.md) | segment reductions, graph pooling, `add_self_loops`, `batch_graphs` |

Graphs are plain arrays in COO format: node features plus `senders`/`receivers` edge indices.

```python
import ion.gnn as gnn

gcn = gnn.GCNConv(16, 32, key=jax.random.key(0))
h = gcn(x, senders, receivers)
```

The [GNN guide](gnn/guide.md) follows messages through a graph and covers the
shared COO, batching, pooling, and self-loop conventions. The [layer
reference](gnn/layers/index.md) compares the available graph convolutions.

## Native transforms

There is no `ion.jit` or `ion.grad`. A model's only pytree leaves are array params; activations and other config are kept as static metadata, meaning `jax.jit`, `jax.grad`, and `jax.vmap` etc. *always* work on Ion modules:

```python
import jax.numpy as jnp

def mse_loss(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

x, y = jnp.ones((32, 4)), jnp.ones((32, 3))

# Loss derivative w.r.t. model parameters
out = jax.jit(model)(x)
grads = jax.grad(mse_loss)(model, x, y)

# Model ensemble
keys = jax.random.split(jax.random.key(0), 8)
ensemble = jax.vmap(MLP)(key=keys)
preds = jax.vmap(lambda m: m(x))(ensemble)
```

## Training example

A training step is standard JAX. The loss takes the model first so `jax.grad` differentiates with respect to it, and the whole step compiles with `jax.jit`:

```python
import optax
import ion

optimizer = ion.Optimizer(optax.adam(3e-4), model)

def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(model, optimizer, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss

for x, y in load_iris():
    model, optimizer, loss = train_step(model, optimizer, x, y)
```

## Checkpointing

`ion.save` persists any pytree to a `.ion` file. `load` takes a reference tree that supplies the structure.

```python
ion.save("model.ion", model)
model = ion.load("model.ion", model)

# Save model and optimizer together to resume training later
ion.save("checkpoint.ion", (model, optimizer))
model, optimizer = ion.load("checkpoint.ion", (model, optimizer))
```

See [Checkpointing](guides/checkpointing.md) for the format and edge cases.

## Module sugar

A `Module` is a plain pytree, but it carries a few conveniences for commonly used operations. Since modules are immutable, anything that transforms the model returns a new one:

```python
model.astype(jnp.bfloat16)                    # cast params to another dtype
model.freeze()                                # freeze every param
model.unfreeze()                              # unfreeze every param
model.at.layer_1.set(model.layer_1.freeze())  # freeze a single submodule
model.at.layer_2.w.set(new_w)                 # replace a leaf deep in the tree
model.at[nn.Dropout].p.set(0.0)               # set a field on every matching layer
```

Read-only introspection returns plain values:

```python
model.num_params                              # total parameter count
model.params                                  # Param leaves, everything else None
```

Casting is how Ion does [mixed precision](guides/mixed-precision.md); see [Freezing](guides/freezing.md) for working with trainability.

## How Ion compares

### Equinox

[Equinox](https://github.com/patrick-kidger/equinox) is a toolkit for scientific computing with pytrees, of which neural networks are one use case. Its API is broader and leans toward flexibility. Ion is narrower, just NN and GNN, preferring ease of use and simplicity.

### Flax NNX

[Flax NNX](https://github.com/google/flax) models are mutable graph objects rather than pytrees, bridged to JAX by custom transforms (`nnx.jit`, `nnx.grad`) and complex machinery that track the graph behind the scenes. The style is close to PyTorch's. Ion stays with plain pytrees and the native JAX transforms, making code easier to reason about.

## Where to go next

- [Core](core/module.md) for how Module, Param, and Optimizer fit together.
- The [NN guide](nn/guide.md) and [GNN guide](gnn/guide.md) for walkthroughs
  and shared conventions.
- [Guides](guides/freezing.md) for training, freezing, mixed precision, and checkpointing.
- [Examples](examples/index.md) for end-to-end projects on real datasets.
