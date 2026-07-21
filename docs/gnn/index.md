# GNN

Graph neural network layers and operations, imported from `ion.gnn`. This page covers the conventions shared across the module: how graphs are represented, self-loops, shape labels, and batching. See the layer pages ([GCN](layers/gcn.md), [GAT](layers/gat.md), [GIN](layers/gin.md)) for the convolutions and [Ops](ops.md) for the segment reductions, pooling, and graph-building helpers.

## Graph Representation

Graphs are represented as plain arrays, no custom graph object:

| Array | Type | Shape | Meaning |
|-------|------|-------|---------|
| `x` | float | `(n, d)` | Node feature matrix (n nodes, d features) |
| `senders` | int | `(e,)` | Source node index for each edge |
| `receivers` | int | `(e,)` | Destination node index for each edge |
| `x_edge` | float | `(e, f)` | Edge feature matrix (optional, GATConv/GATv2Conv only) |

Edges are directed. For undirected graphs, include both directions:

```python
# Triangle: 0-1, 1-2, 0-2 (undirected = 6 directed edges)
senders   = jnp.array([0, 1, 1, 2, 0, 2])
receivers = jnp.array([1, 0, 2, 1, 2, 0])
```

This is a COO (coordinate) sparse format. Storage is O(edges), not O(nodes^2). All operations use `jax.ops.segment_sum` for aggregation, which is JIT-friendly and efficient.

All GNN layers expect unbatched inputs. Passing a batched `(b, n, d)` tensor or a 2D edge index `(e, 2)` will raise immediately via tuple unpacking at the top of each `__call__`. For batching multiple graphs, see [Batching Multiple Graphs](#batching-multiple-graphs) below.

## Self-Loops

The standard GCN formulation (Kipf & Welling, 2017) operates on A_hat = A + I, meaning every node includes its own features in the aggregation. Self-loops are **not** added automatically. Use `add_self_loops` to append them:

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
```

Without self-loops, a node's output depends only on its neighbors, not itself. This is almost never what you want for GCNConv. For GATConv, self-loops allow the node to attend to its own features. GINConv is the exception: do not add self-loops, since a node's own features enter through its `(1 + eps)` term.

## Shape Annotations

| Label | Meaning | Used in |
|-------|---------|---------|
| `n` | number of nodes | everywhere |
| `e` | number of edges | everywhere |
| `g` | number of graphs | mean_pool, sum_pool, max_pool, batch_graphs |
| `d` | node feature dimension | pooling, batch_graphs |
| `i` | input features | GCNConv, GATConv, GATv2Conv, GINConv |
| `o` | output features | GCNConv, GATConv, GATv2Conv, GINConv |
| `h` | number of attention heads | GATConv, GATv2Conv |
| `k` | per-head dimension | GATConv, GATv2Conv |
| `f` | edge feature dimension | GATConv, GATv2Conv (edge_dim) |

## Batching Multiple Graphs

For a batch of graphs with different sizes, the standard approach is to pack them into a single disconnected graph. `batch_graphs` concatenates node features, offsets the edge indices of each graph by the cumulative node count, and returns a `graph_ids` array mapping each node to its source graph:

```python
xs = [x1, x2]                    # (3, d) and (2, d)
senders_list = [s1, s2]
receivers_list = [r1, r2]

x, senders, receivers, graph_ids = gnn.batch_graphs(xs, senders_list, receivers_list)
# x: (5, d), edges of graph 2 offset by 3, graph_ids: [0, 0, 0, 1, 1]
```

Since the subgraphs are disconnected, message passing on the union is exactly equivalent to processing each graph separately. For graph-level predictions, pool node features per graph with `graph_ids`:

```python
h = conv(x, senders, receivers)
y = gnn.mean_pool(h, graph_ids, num_graphs=len(xs))  # (g, d)
```

Call `batch_graphs` outside `jit`: per-batch shapes vary, and each new shape triggers recompilation. For static shapes across batches, pad the batched arrays to a fixed maximum number of nodes and edges. Dummy padding nodes are disconnected (no edges), so they do not affect the output. Mask them out when computing losses or metrics.

## Example

Full examples: [Node classification on Cora](https://github.com/auxeno/ion/blob/main/examples/gnn_cora.py) | [Molecular property prediction on BBBP](https://github.com/auxeno/ion/blob/main/examples/gnn_bbbp.ipynb)

Node classification on a small graph:

```python
import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn, gnn

class NodeClassifier(nn.Module):
    gcn_1: gnn.GCNConv
    gcn_2: gnn.GCNConv

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key):
        key_1, key_2 = jax.random.split(key)
        self.gcn_1 = gnn.GCNConv(in_dim, hidden_dim, key=key_1)
        self.gcn_2 = gnn.GCNConv(hidden_dim, num_classes, key=key_2)

    def __call__(self, x, senders, receivers):
        x = jax.nn.relu(self.gcn_1(x, senders, receivers))
        x = self.gcn_2(x, senders, receivers)
        return x

# Initialize
model = NodeClassifier(16, 32, 7, key=jax.random.key(0))
optimizer = ion.Optimizer(optax.adam(1e-3), model)

# Add self-loops to graph edges
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)
```
