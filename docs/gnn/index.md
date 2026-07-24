# GNN

An ordinary neural network layer transforms each item from its features. A
graph neural network layer also uses connections between items. Ion represents
both parts as plain JAX arrays:

- `x` stores one feature vector per node.
- `senders` and `receivers` store the directed edges.

There is no graph container or custom data type. This page builds one graph
from those three arrays, follows information through it, and then collects the
input conventions shared by every layer in `ion.gnn`.

## Build a Graph

The example has six nodes. Each node starts with an exclusive one-hot feature,
so it is always possible to tell where a contribution originated:

```python
import jax
import jax.numpy as jnp

from ion import gnn

x = jnp.array([
    [1, 0, 0, 0, 0, 0],  # node 0 has feature e_0
    [0, 1, 0, 0, 0, 0],  # node 1 has feature e_1
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=jnp.float32)
```

The rows are nodes and the columns are features:

| Axis | Size | Meaning |
|---|---:|---|
| `x.shape[0]` | 6 | Number of nodes |
| `x.shape[1]` | 6 | Features per node |

The features do not describe the connections. Two nodes can have feature
vectors without being connected, and an edge can connect nodes with completely
different features.

### Add Edges in COO Format

Edges are stored as two parallel one-dimensional arrays:

```python
senders   = jnp.array([0, 1, 0, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 4, 5])
receivers = jnp.array([1, 0, 2, 0, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 4])
```

Read the arrays vertically at the same index:

| `senders[i]` | `receivers[i]` | Meaning |
|---:|---:|---|
| **0** | **1** | **Node 0 sends to node 1** |
| 1 | 0 | Node 1 sends to node 0 |
| 0 | 2 | Node 0 sends to node 2 |
| 2 | 0 | Node 2 sends to node 0 |
| 1 | 3 | Node 1 sends to node 3 |
| 3 | 1 | Node 3 sends to node 1 |

This is coordinate, or COO, sparse format. Each pair
`(senders[i], receivers[i])` is one directed edge. The first two entries encode
the undirected connection `0 <-> 1` as two directed edges:

```python
senders[0], receivers[0]  # 0, 1: node 0 sends to node 1
senders[1], receivers[1]  # 1, 0: node 1 sends to node 0
```

The complete arrays above produce this graph. The highlighted arrow is edge 0.

<iframe
  class="gnn-plot gnn-plot--graph"
  src="../assets/gnn-coo-graph.html"
  title="Six-node graph built from the COO edge arrays"
  loading="lazy"
></iframe>

Edges in Ion are always directed. If the relationship should be undirected,
include both directions as above. COO storage grows with the number of edges,
not with the square of the number of nodes.

## Pass Messages Along Edges

A graph layer uses each directed edge to route information:

1. The sender produces a message.
2. The edge optionally scales or transforms that message.
3. Messages with the same receiver are aggregated.

`senders` has one entry per edge. Indexing `x` with it copies the feature row
of each edge's source node:

```python
messages = x[senders]
messages.shape  # (16 edges, 6 features)

messages[0]  # x[senders[0]] = x[0]
messages[1]  # x[senders[1]] = x[1]
```

Nothing has been combined yet. `messages[i]` is the feature vector travelling
along edge `i`, from `senders[i]` to `receivers[i]`.

`segment_sum` then groups those rows by their receiver. Message `i` is added to
output row `receivers[i]`:

```python
aggregated = gnn.segment_sum(messages, receivers, num_segments=x.shape[0])
```

Node 4 receives edges 6, 8, 12, and 15:

```python
edge_ids = jnp.array([6, 8, 12, 15])

senders[edge_ids]    # [1, 2, 3, 5]
receivers[edge_ids]  # [4, 4, 4, 4]

aggregated[4]  # [0, 1, 1, 1, 0, 1] = x[1] + x[2] + x[3] + x[5]
```

The four messages are grouped into row 4 because their receiver is 4. Node 4's
own feature is absent because the graph does not yet contain a `4 -> 4`
self-loop. This send-then-group operation is the basis of the GCN, GAT, and GIN
layers.

### Include Each Node's Own Features

A node is not automatically its own neighbour. `GCNConv` and the attention
layers normally need self-loop edges so each node can retain its current
features:

```python
num_nodes = x.shape[0]
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)
```

This appends `(0, 0)`, `(1, 1)`, through `(5, 5)`. GIN is the exception: its
`(1 + eps)` term already handles the central node, so do not add self-loops
before `GINConv`.

## Watch Features Mix

A GCN applies a shared linear transformation and then combines features using
degree-normalized edges. To make the graph operation visible by itself, the
plot below fixes the linear transformation to the identity and omits the bias
and activation. It repeatedly applies only the normalized aggregation:

```text
D^-1/2 A D^-1/2 x
```

The input is the one-hot matrix constructed above, with self-loops added.
Choose a step from 0 to 5. Click any node to follow its original feature, and
hover any node to inspect its complete mixed feature vector.

<iframe
  class="gnn-plot gnn-plot--propagation"
  src="../assets/gnn-feature-propagation.html"
  title="One-hot node features mixing over five message-passing steps"
  loading="lazy"
></iframe>

One step lets a feature reach immediate neighbours. By two steps, feature
`e_0` has reached the entire example graph. Later steps continue mixing the
values even though the set of reachable nodes no longer changes.

These are five message-passing steps, which correspond to five GCN layers when
their learned transformations are replaced by the identity. Calling the same
layer five separate times on the original `x` would instead repeat the same
one-step calculation.

## Use a GCN Layer

`GCNConv` learns the feature transformation that the previous plot held fixed:

```python
key = jax.random.key(0)
gcn = gnn.GCNConv(in_dim=6, out_dim=16, key=key)

# (6 nodes, 6 input features) -> (6 nodes, 16 output features)
h = gcn(x, senders, receivers)
```

The same model can stack layers to increase its receptive field:

```python
key_1, key_2 = jax.random.split(key)
gcn_1 = gnn.GCNConv(6, 16, key=key_1)
gcn_2 = gnn.GCNConv(16, 4, key=key_2)

h = jax.nn.relu(gcn_1(x, senders, receivers))
logits = gcn_2(h, senders, receivers)
```

Ion modules are JAX pytrees, so `jax.jit`, `jax.grad`, and the usual Ion
optimizer workflow apply without graph-specific transforms.

## Layer Reference

| Family | Layers | Neighbour aggregation |
|---|---|---|
| [GCN](layers/gcn.md) | `GCNConv` | Fixed symmetric degree normalization |
| [GAT](layers/gat.md) | `GATConv`, `GATv2Conv` | Learned attention weights |
| [GIN](layers/gin.md) | `GINConv` | Sum plus a separate central-node term |
| [Ops](ops.md) | Segment reductions, pooling, graph construction | Functions for custom graph layers |

### Graph Array Contract

| Array | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float | `(n, d)` | Node feature matrix |
| `senders` | int | `(e,)` | Source node for every directed edge |
| `receivers` | int | `(e,)` | Destination node for every directed edge |
| `x_edge` | float | `(e, f)` | Optional edge features for GAT and GATv2 |

All GNN layers accept one graph at a time. A batched `(b, n, d)` node tensor or
an edge matrix shaped `(e, 2)` is not accepted. Pass the two COO edge arrays
separately.

### Self-Loop Reference

| Layer | Add self-loops? | Reason |
|---|---|---|
| `GCNConv` | Normally yes | Include the node's current features in aggregation |
| `GATConv`, `GATv2Conv` | Normally yes | Let a node attend to itself |
| `GINConv` | No | The `(1 + eps)` term already includes the node |

Self-loops are explicit and are never added inside a layer.

### Batch Graphs

Graphs with different numbers of nodes are packed into one disconnected graph:

```python
x, senders, receivers, graph_ids = gnn.batch_graphs(
    [x_1, x_2],
    [senders_1, senders_2],
    [receivers_1, receivers_2],
)
```

`batch_graphs` concatenates the features, offsets the second graph's edge
indices, and records the source graph for every node. Since there are no edges
between the components, message passing is equivalent to processing each graph
separately.

For graph-level outputs, pool the node features by `graph_ids`:

```python
h = gcn(x, senders, receivers)
graph_h = gnn.mean_pool(h, graph_ids, num_graphs=2)
```

Call `batch_graphs` outside `jax.jit`. If batch shapes vary, each new shape
causes a compilation. Pad nodes and edges to fixed maximum sizes when static
shapes are required.

### Shape Labels

| Label | Meaning |
|---|---|
| `n` | Number of nodes |
| `e` | Number of directed edges |
| `g` | Number of graphs |
| `d` | General node feature dimension |
| `i`, `o` | Input and output feature dimensions |
| `h`, `k` | Attention heads and per-head dimension |
| `f` | Edge feature dimension |

Full examples: [node classification on Cora](https://github.com/auxeno/ion/blob/main/examples/gnn_cora.py)
and [molecular property prediction on BBBP](https://github.com/auxeno/ion/blob/main/examples/gnn_bbbp.ipynb).
