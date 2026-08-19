# Overview

Ion introduces four core concepts for building and training neural networks in JAX. Its neural and graph network layers are built on top of them.

## Quickstart

Install Ion with `pip`. For GPU or TPU support, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
pip install ion-nn
```

Construct a model from the built-in layers, then update it with native JAX transformations:

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

The loss takes the model first so `jax.grad` differentiates with respect to it, and the whole step compiles with `jax.jit`. The sections below introduce the four core concepts, network layers, and common workflows.

## The core

Three concepts describe a model; the fourth trains it:

| Concept | Job |
|---|---|
| [**Module**](core/module.md) | Gives the model structure and composes layers. |
| [**Param**](core/param.md) | Marks an array as a trainable or frozen model parameter. |
| [**Buffer**](core/buffers.md) | Holds non-trainable state for stateful layers. |
| [**Optimizer**](core/optimizer.md) | Applies an [Optax](https://github.com/google-deepmind/optax) update to trainable params. |

Built-in layers set up their own params and buffers. You compose them into a model and train it with an optimizer; custom layers declare their own fields.

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

Modules can contain other modules, so layers compose into one model tree. JAX arrays and parameters are dynamic leaves; Python values are compile-time constants, so changing one produces a new JIT specialization.

```text
MLP(  # 131 params, 524 B
  activation=relu, final_activation=None,
  # Modules:
  (0): Linear(  # 80 params, 320 B
    # Parameters:
    w=Param(float32(4, 16)),
    b=Param(float32(16,)),
  ),
  (1): Linear(  # 51 params, 204 B
    # Parameters:
    w=Param(float32(16, 3)),
    b=Param(float32(3,)),
  ),
)
```

In a terminal, the same tree prints as text, colored per layer type. In IPython/Jupyter environments, models render interactively with [Treescope](https://github.com/google-deepmind/treescope).

## Neural network layers

Each [`ion.nn`](nn/layers/index.md) layer is a `Module`, constructed with a `key` for weight initialization:

| Family | Layers |
|--------|--------|
| [Linear](nn/layers/linear.md) | `Linear`, `Identity` |
| [Convolution](nn/layers/conv.md) | `Conv`, `ConvTranspose` |
| [Attention](nn/layers/attention.md) | `MultiHeadAttention` |
| [Normalization](nn/layers/norm.md) | `LayerNorm`, `RMSNorm`, `BatchNorm`, `GroupNorm`, `SpectralNorm` |
| [Recurrent](nn/layers/recurrent.md) | `RNN`, `LSTM`, `GRU` |
| [State Space](nn/layers/ssm.md) | `S4D`, `S5` |
| [Embedding](nn/layers/embedding.md) | `Embedding` |
| [Positional](nn/layers/positional.md) | `RoPE`, `LearnedPositionalEmbedding`, `SinusoidalPositionalEmbedding` |
| [Pooling](nn/layers/pool.md) | `MaxPool`, `AvgPool` |
| [Stochastic](nn/layers/stochastic.md) | `Dropout`, `DropPath` |
| [Blocks](nn/layers/blocks.md) | `MLP`, `Sequential`, `Residual`, `Bidirectional`, `Ensemble` |

```python
import jax
import jax.numpy as jnp

from ion import nn

attn = nn.MultiHeadAttention(64, num_heads=8, key=jax.random.key(0))
y = attn(jnp.ones((32, 16, 64)))
```

The [NN guide](nn/guide.md) builds and trains a model and collects the shared array conventions. The [layer reference](nn/layers/index.md) lists the available families.

## Stateful layers

Some layers, such as `BatchNorm`, update non-trainable values like running statistics during forward passes. Ion stores these in [`Buffer`](core/buffers.md) fields, so stateful layers need no separate state argument or return value.

```python
import jax.numpy as jnp

from ion import nn

model = nn.BatchNorm(64)
x = jnp.ones((8, 64))
y = model(x, training=True)
```

## Graph neural network layers

[`ion.gnn`](gnn/guide.md) provides graph layers and message-passing operations.

| Family | Layers |
|--------|--------|
| [Convolution](gnn/layers/conv.md) | `GCNConv`, `GraphConv`, `SAGEConv` |
| [Attention](gnn/layers/attention.md) | `GATConv`, `GATv2Conv`, `TransformerConv` |
| [Isomorphism](gnn/layers/isomorphism.md) | `GINConv`, `GINEConv` |
| [Composite](gnn/layers/composite.md) | `GraphNetwork`, `EdgeUpdate`, `NodeUpdate` |
| [Relational](gnn/layers/relational.md) | `RGCNConv`, `HGTConv` |
| [Gated](gnn/layers/gated.md) | `GatedGCNConv` |
| [Pooling](gnn/layers/pool.md) | `GlobalAttentionPool`, `MultiHeadAttentionPool` |
| [Operations](gnn/operations.md) | segment reductions, pooling, topology, batching |

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

The [GNN guide](gnn/guide.md) follows messages through a graph and covers the shared COO, batching, pooling, and self-loop conventions. The [layer reference](gnn/layers/index.md) compares the available graph convolutions.

## Native transforms

There is no `ion.jit` or `ion.grad`. Modules are native pytrees: parameters and ordinary array fields are dynamic, while activations and other configuration are static metadata. Stateless model calls therefore work directly with `jax.jit`, `jax.grad`, `jax.vmap`, and other JAX transforms:

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
ensemble = nn.Ensemble(
    lambda key: nn.MLP([4, 16, 3], key=key),
    8,
    key=jax.random.key(0),
)
preds = ensemble(x)
```

Stateful calls work directly with `jax.jit`, `jax.grad`, and `jax.lax.scan` too. Writing a shared buffer under a parallelizing transform such as `jax.vmap`, or inside `jax.checkpoint`, has additional constraints; see [Sharp edges](sharp-edges.md#buffer-mutation-and-jax-transforms).

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

model.astype(jnp.bfloat16)  # cast params to another dtype
model.freeze()  # freeze every param
model.unfreeze()  # unfreeze every param
model.at.layers[0].set(model.layers[0].freeze())  # freeze a single submodule
model.at.layers[1].w.set(new_w)  # replace a leaf deep in the tree

dropout_model = nn.Sequential(nn.Dropout(0.1))
dropout_model.at[nn.Dropout].p.set(0.0)  # set every matching layer
```

Read-only introspection returns plain values:

```python
import jax

from ion import nn

model = nn.MLP([4, 16, 3], key=jax.random.key(0))

model.num_params  # total parameter count
model.disk_usage  # size of the arrays a checkpoint would hold, e.g. '524 B'
model.params  # Param leaves; array data and buffers become None
```

Casting is how Ion does [mixed precision](workflows.md#mixed-precision); see [Freezing](workflows.md#freezing) for working with trainability.

## Benchmarks

Ion has been benchmarked against Equinox, Flax NNX, and PyTorch across MLP, ResNet, and GPT workloads on an NVIDIA H100.

Ion performs on par with the fastest of the compared frameworks. See [Benchmarks](benchmarks.md) for plots, scope, and methodology.

## How Ion compares

### Equinox

[Equinox](https://github.com/patrick-kidger/equinox) is a general scientific-computing toolkit with neural networks as one use case. Like Ion, it is pytree-first, works with ordinary JAX arrays and explicit random keys, and remains compatible with the wider JAX ecosystem. Its larger, filtering-based API provides transforms such as `eqx.filter_jit` and utilities such as `filter`, `partition`, and `combine` for controlling arbitrary pytree leaves. Ion is focused specifically on neural and graph networks and records the roles commonly needed for them through `Param`, `Buffer`, and `Optimizer`. This allows models to use native JAX transforms directly while Ion automatically handles trainability, optimizer partitioning, and layer state, producing a smaller, more consistent, and less verbose API at the expense of Equinox's fine-grained per-operation control.

### Flax NNX

[Flax NNX](https://github.com/google/flax) is the current neural network API in Flax, a JAX ecosystem maintained by Google DeepMind. It evolved from Linen as a simplified API built around Python reference semantics, so models are mutable object graphs supported by abstractions such as `Variable`, `State`, `GraphDef`, filters, `split` and `merge`, graph-aware transforms such as `nnx.jit`, `nnx.grad`, and `nnx.vmap`, and a more PyTorch-like mutable `Rngs` system. This supports shared references and general mutable state, but introduces a substantial API and machinery around JAX. Ion keeps models as ordinary immutable JAX pytrees, with `Buffer` for stateful layers as a narrow exception, passes random keys explicitly, and works directly with `jax.jit`, `jax.grad`, and `jax.vmap`. Its four-concept core supports both neural and graph networks through a smaller, more uniform user-facing API that stays closer to JAX's functional model and is easier to reason about.

## Where to go next

- [Core](core/module.md) for how Module, Param, Buffer, and Optimizer fit together.
- The [NN guide](nn/guide.md) and [GNN guide](gnn/guide.md) for walkthroughs and shared conventions.
- [Workflows](workflows.md) for freezing, mixed precision, and checkpointing.
- [Sharp edges](sharp-edges.md) for known constraints and gotchas.
- [Examples](examples/index.md) for end-to-end projects on real datasets.
