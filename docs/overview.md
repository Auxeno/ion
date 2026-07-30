# Overview

Ion introduces three core abstractions for building and training neutral networks in JAX. Its neural and graph network layers are built on top of them.

## Quickstart

Install Ion with `pip`. For GPU or TPU support, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
pip install ion-nn
```

Construct a model from the built-in layers, then update it with native JAX
transformations:

```python
import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn

# Four input features, three outputs
model = nn.Linear(4, 3, key=jax.random.key(0))

x = jnp.ones((32, 4))
y = jnp.ones((32, 3))

def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

optimizer = ion.Optimizer(optax.adam(1e-2), model)

@jax.jit
def train_step(model, optimizer, x, y):
    grads = jax.grad(loss_fn)(model, x, y)
    return optimizer.update(model, grads)

model, optimizer = train_step(model, optimizer, x, y)
```

The loss takes the model first so `jax.grad` differentiates with respect to it, and the whole step compiles with `jax.jit`. The sections below introduce the three core abstractions, network layers and common workflows.

## The core

- [**Param**](core/param.md) wraps a JAX array and marks it trainable or frozen.
- [**Module**](core/module.md) is the base class for models and layers. A model is an immutable [pytree](https://docs.jax.dev/en/latest/pytrees.html): params and submodules at the leaves, everything else in the structure.
- [**Optimizer**](core/optimizer.md) wraps any [optax](https://github.com/google-deepmind/optax) transform and updates a model, automatically partitioning out non-trainable parameters.

The whole core is under a thousand lines of code, small enough to read in an afternoon.

## Building a model

Subclass `Module`, declare fields as class annotations, and assign them in `__init__`:

```python
import typing

import jax
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

```text
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
| [Linear](nn/layers/linear.md) | `Linear` |
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
| [Identity](nn/layers/identity.md) | `Identity` |
| [Composite](nn/layers/mlp.md) | `MLP`, `Sequential` |

```python
import jax
import jax.numpy as jnp

from ion import nn

attn = nn.SelfAttention(64, num_heads=8, key=jax.random.key(0))
y = attn(jnp.ones((32, 16, 64)))
```

The [NN guide](nn/guide.md) builds and trains a model and collects the shared
array conventions. The [layer reference](nn/layers/index.md) lists the
available families.

## Graph neural network layers

[`ion.gnn`](gnn/guide.md) provides graph layers and message-passing operations.

| Family | Layers |
|--------|--------|
| [Graph Convolution](gnn/layers/gcn.md) | `GCNConv`, `GraphConv` |
| [Graph Attention](gnn/layers/gat.md) | `GATConv`, `GATv2Conv`, `TransformerConv` |
| [Graph Isomorphism Network](gnn/layers/gin.md) | `GINConv` |
| [GraphSAGE](gnn/layers/sage.md) | `SAGEConv` |
| [Operations](gnn/operations.md) | segment reductions, graph pooling, `add_self_loops`, `batch_graphs` |

Graphs are plain arrays in COO format: node features plus `senders`/`receivers` edge indices.

```python
import jax
import jax.numpy as jnp

from ion import gnn

x = jnp.ones((3, 16))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
gcn = gnn.GCNConv(16, 32, key=jax.random.key(0))
h = gcn(x, senders, receivers)
```

The [GNN guide](gnn/guide.md) follows messages through a graph and covers the
shared COO, batching, pooling, and self-loop conventions. The [layer
reference](gnn/layers/index.md) compares the available graph convolutions.

## Native transforms

There is no `ion.jit` or `ion.grad`. A model's only pytree leaves are array params; activations and other config are kept as static metadata, meaning `jax.jit`, `jax.grad`, and `jax.vmap` etc. *always* work on Ion modules:

```python
import jax
import jax.numpy as jnp

from ion import nn

model = nn.MLP([4, 16, 3], key=jax.random.key(0))

def mse_loss(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

x, y = jnp.ones((32, 4)), jnp.ones((32, 3))

# Loss derivative w.r.t. model parameters
out = jax.jit(model)(x)
grads = jax.grad(mse_loss)(model, x, y)

# Model ensemble
keys = jax.random.split(jax.random.key(0), 8)
ensemble = jax.vmap(lambda key: nn.MLP([4, 16, 3], key=key))(keys)
preds = jax.vmap(lambda m: m(x))(ensemble)
```

## Checkpointing

`ion.save` persists any pytree to a `.ion` file. `load` takes a reference tree that supplies the structure.

```python
import jax
import optax

import ion
from ion import nn

model = nn.Linear(4, 3, key=jax.random.key(0))
optimizer = ion.Optimizer(optax.adam(3e-4), model)

ion.save("model.ion", model)
model = ion.load("model.ion", model)

# Save model and optimizer together to resume training later
ion.save("checkpoint.ion", (model, optimizer))
model, optimizer = ion.load("checkpoint.ion", (model, optimizer))
```

See [Checkpointing](workflows.md#checkpointing) for the format and edge cases.

## Module helpers

A `Module` is a plain pytree, but it carries a few conveniences for commonly used operations. Since modules are immutable, anything that transforms the model returns a new one:

```python
import jax
import jax.numpy as jnp

from ion import nn

model = nn.MLP([4, 16, 3], key=jax.random.key(0))
new_w = jnp.zeros_like(model.layers[1].w)

model.astype(jnp.bfloat16)                         # cast params to another dtype
model.freeze()                                     # freeze every param
model.unfreeze()                                   # unfreeze every param
model.at.layers[0].set(model.layers[0].freeze())   # freeze a single submodule
model.at.layers[1].w.set(new_w)                    # replace a leaf deep in the tree

dropout_model = nn.Sequential(nn.Dropout(0.1))
dropout_model.at[nn.Dropout].p.set(0.0)             # set every matching layer
```

Read-only introspection returns plain values:

```python
import jax

from ion import nn

model = nn.MLP([4, 16, 3], key=jax.random.key(0))

model.num_params                              # total parameter count
model.params                                  # Param leaves, everything else None
```

Casting is how Ion does [mixed precision](workflows.md#mixed-precision); see
[Freezing](workflows.md#freezing) for working with trainability.

## Benchmarks

Ion has been benchmarked against Equinox, Flax NNX, and PyTorch across MLP, ResNet, and GPT workloads on an NVIDIA H100. 

Results show Ion generally has the best performance. See [Benchmarks](benchmarks.md) for plots and more details.

## How Ion compares

### Equinox

[Equinox](https://github.com/patrick-kidger/equinox) is a toolkit for scientific computing with pytrees, of which neural networks are one use case. Its API is broader and leans toward flexibility. Ion is narrower, just NN and GNN, preferring ease of use and simplicity.

### Flax NNX

[Flax NNX](https://github.com/google/flax) models are mutable graph objects rather than pytrees, bridged to JAX by custom transforms (`nnx.jit`, `nnx.grad`) and complex machinery that track the graph behind the scenes. The style is close to PyTorch's. Ion stays with plain pytrees and the native JAX transforms, making code easier to reason about.

## Where to go next

- [Core](core/module.md) for how Module, Param, and Optimizer fit together.
- The [NN guide](nn/guide.md) and [GNN guide](gnn/guide.md) for walkthroughs
  and shared conventions.
- [Workflows](workflows.md) for freezing, mixed precision, and checkpointing.
- [Sharp edges](sharp-edges.md) for known constraints and gotchas.
- [Examples](examples/index.md) for end-to-end projects on real datasets.
