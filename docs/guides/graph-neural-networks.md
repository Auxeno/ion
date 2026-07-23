# Graph Neural Networks

A walkthrough of building graph data and training a GNN with `ion.gnn`. For the array format, self-loops, and shape labels in full, see the [GNN Reference](../gnn/reference.md); for the individual convolutions see [Layers](../gnn/index.md).

## Build a graph

A graph is plain arrays, with no custom object to construct. Node features live in `x`, and each edge is a pair of entries in `senders`/`receivers` (a COO sparse format). Edges are directed, so an undirected graph lists both directions:

```python
import jax.numpy as jnp

# Triangle over 3 nodes: 0-1, 1-2, 0-2 (undirected = 6 directed edges)
x         = jnp.ones((3, 16))                  # (n, d) node features
senders   = jnp.array([0, 1, 1, 2, 0, 2])
receivers = jnp.array([1, 0, 2, 1, 2, 0])
```

## Add self-loops

Self-loops are not added automatically. Most convolutions need them so a node includes its own features in the aggregation:

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes=3)
```

Add them for `GCNConv` and `GATConv`, but not for `GINConv`, which folds a node's own features in through its `(1 + eps)` term. See the [Reference](../gnn/reference.md#self-loops) for why.

## Run a layer

A convolution is a [`Module`](../core/module.md) like any other. It takes the node features and the edge indices, and returns updated node features:

```python
from ion import gnn
import jax

gcn = gnn.GCNConv(in_dim=16, out_dim=32, key=jax.random.key(0))
h = gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
```

GNN layers expect unbatched inputs: a single `(n, d)` feature matrix and 1D edge indices.

## Train it

Training is identical to the [NN case](neural-networks.md): the same `jax.grad` and [`Optimizer`](../core/optimizer.md). Here is a two-layer node classifier:

```python
import optax
import ion
import ion.nn as nn

class GCN(nn.Module):
    conv_1: gnn.GCNConv
    conv_2: gnn.GCNConv

    def __init__(self, in_dim, hidden, out_dim, *, key):
        keys = jax.random.split(key, 2)
        self.conv_1 = gnn.GCNConv(in_dim, hidden, key=keys[0])
        self.conv_2 = gnn.GCNConv(hidden, out_dim, key=keys[1])

    def __call__(self, x, senders, receivers):
        h = jax.nn.relu(self.conv_1(x, senders, receivers))
        return self.conv_2(h, senders, receivers)

def loss_fn(model, x, senders, receivers, y, mask):
    logits = model(x, senders, receivers)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return (losses * mask).sum() / mask.sum()

@jax.jit
def train_step(model, optimizer, x, senders, receivers, y, mask):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, senders, receivers, y, mask)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss

model = GCN(in_dim=16, hidden=64, out_dim=7, key=jax.random.key(0))
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

The `mask` restricts the loss to the training nodes, the standard setup for transductive node classification.

## Batch multiple graphs

For graph-level tasks, combine several graphs into one big disconnected graph with `batch_graphs`, then read out per-graph with the pooling ops:

```python
from ion.gnn import batch_graphs

x, senders, receivers, batch = batch_graphs(xs, senders_list, receivers_list)
```

`batch` labels each node with its graph index, which the graph-level [pooling ops](../gnn/ops.md) use to reduce node features to one vector per graph.

## Next

- [GNN Reference](../gnn/reference.md) for the array format and batching in detail.
- [Ops](../gnn/ops.md) for segment reductions, pooling, and graph builders.
- [Examples](../examples/index.md) for node classification on Cora and molecular property prediction on BBBP.
