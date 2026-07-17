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

All GNN layers expect unbatched inputs. Passing a batched `(b, n, d)` tensor or a 2D edge index `(e, 2)` will raise immediately via tuple unpacking at the top of each `__call__`. For batching multiple graphs, see [Batching Multiple Graphs](#batching-multiple-graphs) below.

## Self-Loops

The standard GCN formulation (Kipf & Welling, 2017) operates on A_hat = A + I, meaning every node includes its own features in the aggregation. Self-loops are **not** added automatically. Use `add_self_loops` to append them:

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
```

Without self-loops, a node's output depends only on its neighbors, not itself. This is almost never what you want for GCNConv. For GATConv, self-loops allow the node to attend to its own features. GINConv is the exception: do not add self-loops, since a node's own features enter through its `(1 + eps)` term.

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

**Edge masking.** Pass a boolean `edge_mask` of shape `(e,)` to disable individual edges. Edges marked `False` get `-inf` attention logits (zero attention weight) and their edge features are zeroed:

```python
y = gat(x, senders, receivers, x_edge, edge_mask)  # edge_mask shape: (e,) bool
```

This is useful for padded batches (mask out dummy edges) or dropping edges at inference without rebuilding the edge index.

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

**Edge features.** Same `edge_dim` / `x_edge` / `edge_mask` interface as GATConv (`False` edges get zero attention). The difference is that edge features are added *inside* the LeakyReLU (before the attention dot product), so the nonlinearity mixes node and edge information:

```python
gat = gnn.GATv2Conv(in_dim=16, out_dim=32, num_heads=4, edge_dim=8, key=key)
y = gat(x, senders, receivers, x_edge)  # x_edge shape: (e, 8)
```

### GINConv

Graph Isomorphism Network (Xu et al., 2019). Sum-aggregates neighbor features and applies an MLP to `(1 + eps) * x + aggregated`. Sum aggregation preserves neighbor multiplicity, making GIN as discriminative as the Weisfeiler-Lehman graph isomorphism test.

```python
from ion import nn, gnn

gin = gnn.GINConv(nn.MLP([16, 32, 32], key=key))
y = gin(x, senders, receivers)  # (n, 16) -> (n, 32)
```

The update network is supplied by the caller, so `GINConv` takes no `key` and creates no weights of its own. `eps` weights a node's own features against its aggregated neighbors. It defaults to a fixed `0.0`; set `train_eps=True` to make it a learnable scalar:

```python
gin = gnn.GINConv(nn.MLP([16, 32, 32], key=key), train_eps=True)
```

Do not add self-loops: own features enter through the `(1 + eps)` term.

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

GINConv creates no weights of its own; initialization is determined by the update network the caller supplies. With `train_eps=True` its only parameter is the scalar `eps`, initialized to the `eps` argument (default `0.0`).

## Operations

### segment_softmax

Softmax normalized within segments. Used internally by GATConv and GATv2Conv to normalize attention weights per receiver node, but useful for custom GNN layers too.

```python
from ion.gnn import segment_softmax

# Normalize scores so they sum to 1 per receiver node
weights = segment_softmax(scores, receivers, num_nodes)
```

### segment_mean

Mean of data within each segment. JAX ships `segment_sum`, `segment_max`, `segment_min` and `segment_prod` but no mean; this fills the gap. Empty segments give zeros, not NaN.

```python
from ion.gnn import segment_mean

means = segment_mean(messages, receivers, num_nodes)
```

The four `jax.ops` segment reductions are also re-exported from `ion.gnn`, so every segment reduction is reachable as `gnn.segment_*` alongside `segment_softmax` and `segment_mean`.

### Pooling: mean_pool, sum_pool, max_pool

Graph-level readout for graph classification. Each pools node features `(n, d)` into per-graph vectors `(g, d)` using a `graph_ids` array that maps each node to its graph (see [Batching Multiple Graphs](#batching-multiple-graphs) below).

```python
from ion.gnn import mean_pool

g = mean_pool(x, graph_ids, num_graphs)  # (n, d) -> (g, d)
```

`max_pool` returns zeros for empty graphs rather than `segment_max`'s `-inf` fill; `mean_pool` likewise guards empty graphs to zeros. Sum pooling is the readout used in the GIN paper, since it preserves node counts.

### add_self_loops

Appends self-loop edges (i -> i) for every node.

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
# senders and receivers now have num_nodes extra entries
```

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

## Examples

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
