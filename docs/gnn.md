# Graph Neural Networks

Conventions and usage for Ion's graph neural network layers.

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

All GNN layers expect unbatched inputs. Passing a batched `(b, n, d)` tensor or a 2D edge index `(e, 2)` will raise immediately via tuple unpacking at the top of each `__call__`. For batching multiple graphs, see the section below on concatenating disconnected subgraphs.

## Self-Loops

The standard GCN formulation (Kipf & Welling, 2017) operates on A_hat = A + I, meaning every node includes its own features in the aggregation. Self-loops are **not** added automatically. Use `add_self_loops` to append them:

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
```

Without self-loops, a node's output depends only on its neighbors, not itself. This is almost never what you want for GCNConv. For GATConv, self-loops allow the node to attend to its own features.

## Layers

### GCNConv

Graph Convolutional Network (Kipf & Welling, 2017). Applies a shared linear transform then aggregates with symmetric degree normalization: D^{-1/2} A D^{-1/2} X W.

```python
from ion import gnn

gcn = gnn.GCNConv(in_dim=16, out_dim=32, key=key)
y = gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
```

No activation is included. Compose with `jax.nn.relu` or similar:

```python
x = jax.nn.relu(gcn_1(x, senders, receivers))
x = gcn_2(x, senders, receivers)
```

### GATConv

Graph Attention Network (Velickovic et al., 2018). Learns attention weights over each node's neighborhood using LeakyReLU-gated additive attention. Multi-head attention is supported; heads are concatenated.

```python
gat = gnn.GATConv(in_dim=16, out_dim=32, num_heads=4, key=key)
y = gat(x, senders, receivers)  # (n, 16) -> (n, 32)
```

`out_dim` must be divisible by `num_heads`. Each head produces `out_dim // num_heads` features, concatenated to `out_dim`.

**Edge features.** Set `edge_dim` to incorporate per-edge features into attention scores. When provided, edge features are projected into the multi-head space and added to the attention logits before the LeakyReLU gate:

```python
gat = gnn.GATConv(in_dim=16, out_dim=32, num_heads=4, edge_dim=8, key=key)
y = gat(x, senders, receivers, x_edge)  # x_edge shape: (e, 8)
```

When `edge_dim` is None (default), no extra parameters are created and behavior is identical to the standard GATConv. If `edge_dim` is set but `x_edge` is not passed at call time, the edge path is skipped. Passing `x_edge` without setting `edge_dim` will raise an error.

### GATv2Conv

Dynamic Graph Attention Network (Brody et al., 2022). Fixes a theoretical limitation of GATConv where attention rankings are "static" (identical for all query nodes). GATv2 applies LeakyReLU *after* combining sender and receiver features, making attention scores depend on both nodes:

```
GATv1: e_ij = LeakyReLU(a_l^T W h_i + a_r^T W h_j) - static attention
GATv2: e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)     - dynamic attention
```

The interface is identical to GATConv:

```python
gat = gnn.GATv2Conv(in_dim=16, out_dim=32, num_heads=4, key=key)
y = gat(x, senders, receivers)  # (n, 16) -> (n, 32)
```

Structural differences from GATConv: two weight matrices (`w_sender`, `w_receiver`) instead of one, and a single attention vector (`att`) instead of two. This means attention must be computed per-edge rather than decomposed to node-level scores.

**Edge features.** Same `edge_dim` / `x_edge` interface as GATConv. The difference is that edge features are added *inside* the LeakyReLU (before the attention dot product), so the nonlinearity mixes node and edge information:

```python
gat = gnn.GATv2Conv(in_dim=16, out_dim=32, num_heads=4, edge_dim=8, key=key)
y = gat(x, senders, receivers, x_edge)  # x_edge shape: (e, 8)
```

## Shape Annotations

| Label | Meaning | Used in |
|-------|---------|---------|
| `n` | number of nodes | everywhere |
| `e` | number of edges | everywhere |
| `i` | input features | GCNConv, GATConv, GATv2Conv |
| `o` | output features | GCNConv, GATConv, GATv2Conv |
| `h` | number of attention heads | GATConv, GATv2Conv |
| `k` | per-head dimension | GATConv, GATv2Conv |
| `f` | edge feature dimension | GATConv, GATv2Conv (edge_dim) |

## Weight Initialization

| Layer | Weights | Bias |
|-------|---------|------|
| GCNConv | He normal | zeros |
| GATConv (projection) | Glorot uniform | zeros |
| GATConv (attention) | Glorot uniform | - |
| GATConv (edge projection) | Glorot uniform | - |
| GATConv (edge attention) | Glorot uniform | - |
| GATv2Conv (projection) | Glorot uniform | zeros |
| GATv2Conv (attention) | Glorot uniform | - |
| GATv2Conv (edge projection) | Glorot uniform | - |

GCNConv defaults to He normal, matching `Linear`, since it is typically followed by ReLU. GATConv and GATv2Conv use Glorot uniform (activation-agnostic) since the projections feed into a LeakyReLU attention mechanism.

## Operations

### segment_softmax

Softmax normalized within segments. Used internally by GATConv and GATv2Conv to normalize attention weights per receiver node, but useful for custom GNN layers too.

```python
from ion.gnn import segment_softmax

# Normalize scores so they sum to 1 per receiver node
weights = segment_softmax(scores, receivers, num_nodes)
```

### add_self_loops

Appends self-loop edges (i -> i) for every node.

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
# senders and receivers now have num_nodes extra entries
```

## Batching Multiple Graphs

Ion does not provide a graph batching utility. For a batch of graphs with different sizes, the standard approach is to concatenate them into a single disconnected graph and offset the edge indices:

```python
# Graph 1: 3 nodes, edges (0->1, 1->2)
# Graph 2: 2 nodes, edges (0->1)
x = jnp.concatenate([x1, x2])              # (5, d)
senders = jnp.concatenate([s1, s2 + 3])    # offset by num_nodes_1
receivers = jnp.concatenate([r1, r2 + 3])  # offset by num_nodes_1
```

Since the two subgraphs are disconnected, GNN layers process them independently. For graph-level predictions, aggregate node features per graph using `jax.ops.segment_sum` with a graph membership array.

For JIT compatibility, pad the concatenated arrays to a fixed maximum number of nodes and edges so the shapes remain static across batches. Dummy padding nodes are disconnected (no edges), so they do not affect the output. Mask them out when computing losses or metrics.

## Example

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
