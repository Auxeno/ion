# Graph neural networks

This guide represents a graph with JAX arrays, follows messages through it, and
builds node-level and graph-level predictors. For individual graph layers and
their APIs, see the [GNN layer reference](layers/index.md).

Follow the guide from the beginning for a walkthrough, or jump directly to a
topic:

- [COO format](#coo-format)
- [Message passing](#message-passing)
- [Aggregating messages](#aggregating-messages)
- [Self-loops](#self-loops)
- [Feature propagation](#feature-propagation)
- [Building a GNN](#building-a-gnn)
- [Batching graphs](#batching-graphs)
- [Graph pooling](#graph-pooling)
- [Static shapes](#static-shapes)
- [Further examples](#further-examples)

## COO format

A graph starts with an ordinary array of feature vectors. Each row of `x`
represents one node. The example has six nodes with exclusive one-hot features,
so it is always possible to tell where a contribution originated:

```python
import jax
import jax.numpy as jnp

from ion import gnn, nn

x = jnp.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # node 0 has feature e_0
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # node 1 has feature e_1
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
], dtype=jnp.float32)
```

The rows are nodes and the columns are features, so `x` has shape
`(num_nodes, node_dim)`, here `(6, 6)`. An ordinary neural network layer can
transform these rows independently. A graph layer additionally uses edges to
exchange information between connected rows.

Edges are stored in two parallel one-dimensional arrays:

```python
senders   = jnp.array([0, 1, 0, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 4, 5])
receivers = jnp.array([1, 0, 2, 0, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 4])
```

The arrays have shape `(num_edges,)`. Read matching entries at the same index
as one directed edge:

| `senders[i]` | `receivers[i]` | Meaning |
|---:|---:|---|
| **0** | **1** | **Node 0 sends to node 1** |
| 1 | 0 | Node 1 sends to node 0 |
| 0 | 2 | Node 0 sends to node 2 |
| 2 | 0 | Node 2 sends to node 0 |
| 1 | 3 | Node 1 sends to node 3 |
| 3 | 1 | Node 3 sends to node 1 |

This is coordinate, or COO, sparse format. The first two entries encode the
undirected connection `0 <-> 1` as two directed edges:

```python
senders[0], receivers[0]  # 0, 1: node 0 sends to node 1
senders[1], receivers[1]  # 1, 0: node 1 sends to node 0
```

The complete arrays above produce this graph. The highlighted arrow is edge 0.

<iframe
  class="gnn-plot gnn-plot--graph"
  src="../../assets/gnn-coo-graph.html"
  title="Six-node graph built from the COO edge arrays"
  loading="lazy"
></iframe>

Edges in COO are always directed. If the relationship should be undirected,
include both directions as above.

### Edge features

`GraphConv` can apply one scalar weight per directed edge:

```python
edge_weight.shape  # (num_edges,)
h = conv(x, senders, receivers, edge_weight=edge_weight)
```

`GATConv` and `GATv2Conv` can instead include one feature row per directed edge:

```python
x_edge.shape  # (num_edges, edge_features)
h = gat(x, senders, receivers, x_edge=x_edge)
```

Construct the layer with the matching `edge_dim`. If self-loops are required,
the edge-feature array must also contain features for those appended edges. See
the [Graph Attention reference](layers/gat.md) for the complete call contract.

The complete graph representation consists of JAX arrays:

| Array | Shape | Meaning |
|---|---|---|
| `x` | `(num_nodes, node_dim)` | Feature vector for each node |
| `senders` | `(num_edges,)` | Source node of each directed edge |
| `receivers` | `(num_edges,)` | Destination node of each directed edge |
| `edge_weight` | `(num_edges,)` | Optional scalar weight for each edge |
| `x_edge` | `(num_edges, edge_dim)` | Optional feature vector for each edge |

## Message passing

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

## Aggregating messages

`segment_sum` groups message rows by their receiver. Message `i` is added to
output row `receivers[i]`:

```python
aggregated = gnn.segment_sum(messages, receivers, num_segments=x.shape[0])
```

Node 4 receives edges 6, 8, 12, and 15:

```python
edge_ids = jnp.array([6, 8, 12, 15])

senders[edge_ids]    # [1, 2, 3, 5]
receivers[edge_ids]  # [4, 4, 4, 4]

aggregated[4]  # [0.0, 1.0, 1.0, 1.0, 0.0, 1.0] = x[1] + x[2] + x[3] + x[5]
```

The four messages are grouped into row 4 because their receiver is 4. Node 4's
own feature is absent because the graph does not yet contain a `4 -> 4`
self-loop. This send-then-group operation is the basis of the GCN, GraphConv,
GAT, GIN, and GraphSAGE layers.

## Self-loops

A node is not automatically its own neighbour. `GCNConv` and the attention
layers normally need self-loop edges so each node can retain its current
features:

```python
num_nodes = x.shape[0]
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)
```

This appends `(0, 0)`, `(1, 1)`, through `(5, 5)`. Self-loops are explicit and
are never added inside a layer.

| Layer | Add self-loops? | Reason |
|---|---|---|
| `GCNConv` | Normally yes | Include the node's current features in aggregation |
| `GraphConv` | No | The separate root weight already includes the node |
| `GATConv`, `GATv2Conv` | Normally yes | Let a node attend to itself |
| `GINConv` | No | The `(1 + eps)` term already includes the node |
| `SAGEConv` | No | The root weight already includes the node |

`add_self_loops` appends one loop for every node. It does not check whether the
input already contains self-loops, so avoid calling it twice on the same edge
arrays.

## Feature propagation

A GCN applies a shared linear transformation and then combines features using
degree-normalized edges. To make the graph operation visible by itself, the
plot below fixes the linear transformation to the identity and omits the bias
and activation. It repeatedly applies only the normalized aggregation:

\[
D^{-1/2} A D^{-1/2} x
\]

The input is the one-hot matrix constructed above, with self-loops added.
Choose a step from 0 to 5. Click any node to follow its original feature, and
hover any node to inspect its complete mixed feature vector.

<iframe
  class="gnn-plot gnn-plot--propagation"
  src="../../assets/gnn-feature-propagation.html"
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

## Building a GNN

The previous plot fixed the feature transformation to the identity so only the
graph operation was visible. `GCNConv` restores that transformation and makes it
learnable: each layer multiplies the node features by a weight matrix, then
applies the same normalized aggregation as the plot. Training adjusts the weight
matrix, so the layer learns which projection of the input features to mix across
edges:

```python
key = jax.random.key(0)
gcn = gnn.GCNConv(in_dim=6, out_dim=16, key=key)

# (6 nodes, 6 input features) -> (6 nodes, 16 output features)
h = gcn(x, senders, receivers)
```

Each layer performs one message-passing step. Stacking layers lets information
travel across progressively longer paths:

```python
key_gcn_1, key_gcn_2, key_classifier = jax.random.split(key, 3)
gcn_1 = gnn.GCNConv(6, 16, key=key_gcn_1)
gcn_2 = gnn.GCNConv(16, 16, key=key_gcn_2)
classifier = nn.Linear(16, 3, key=key_classifier)

h = jax.nn.relu(gcn_1(x, senders, receivers))
h = jax.nn.relu(gcn_2(h, senders, receivers))
logits = classifier(h)

logits.shape  # (6 nodes, 3 classes)
```

The linear classifier acts on each node independently after the graph layers
have mixed information between connected nodes. Ion modules are JAX
[pytrees](https://docs.jax.dev/en/latest/pytrees.html), so
`jax.jit`, `jax.grad`, and the usual Ion optimizer workflow apply without
graph-specific transforms.

## Batching graphs

Ordinary neural network inputs are often stacked along a leading batch
dimension:

```python
x.shape  # (batch, items, features)
```

Graphs frequently contain different numbers of nodes and edges, so they cannot
be stacked this way without padding. Ion instead concatenates them into one
disconnected graph:

```python
x, senders, receivers, graph_ids = gnn.batch_graphs(
    [x_1, x_2],
    [senders_1, senders_2],
    [receivers_1, receivers_2],
)
```

The packed arrays have shapes:

```python
x.shape          # (total_nodes, node_dim)
senders.shape    # (total_edges,)
receivers.shape  # (total_edges,)
graph_ids.shape  # (total_nodes,)
```

`batch_graphs` offsets the edge indices of each graph and records which graph
each node belongs to in `graph_ids`. There are no edges between the packed
graphs, so message passing cannot mix information between examples. Applying a
graph layer to the disconnected graph is equivalent to applying it to each
graph separately:

```python
h = gcn(x, senders, receivers)
```

## Graph pooling

Node-level tasks use one output row per node. Graph-level tasks reduce node
representations into one row per graph:

```python
graph_h = gnn.mean_pool(h, graph_ids, num_graphs=2)
logits = graph_classifier(graph_h)

graph_h.shape  # (2, hidden_dim)
logits.shape   # (2, num_classes)
```

`graph_ids` serves the role normally played by a batch index, allowing the
packed node representations to be reduced back to one row per graph.

Ion provides three graph readouts:

| Operation | Reduction |
|---|---|
| `mean_pool` | Mean node representation |
| `sum_pool` | Sum node representation |
| `max_pool` | Maximum node representation |

Sum pooling preserves graph-size information, while mean pooling normalizes it
away. The [Operations reference](operations.md) documents the complete pooling
APIs.

## Static shapes

Call `batch_graphs` outside `jax.jit`. It accepts Python sequences of
differently shaped arrays and constructs the packed arrays that enter the
compiled model.

JAX compiles a function for the shapes it receives. If the packed number of
nodes or edges changes, the compiled function may be retraced for the new
shapes. Pad nodes and edges to fixed maximum sizes when a workload requires
static shapes.

### Shape labels

| Label | Meaning |
|---|---|
| `n` | Number of nodes |
| `e` | Number of directed edges |
| `g` | Number of graphs |
| `d` | General node feature dimension |
| `i`, `o` | Input and output feature dimensions |
| `h`, `k` | Attention heads and per-head dimension |
| `f` | Edge feature dimension |

Each layer page defines how these labels apply to its inputs, parameters, and
outputs.

## Further examples

- [GNN on Cora](../examples/gnn-cora.md) performs node classification on one
  citation graph.
- [GNN on BBBP](../examples/gnn-bbbp.ipynb) batches molecular graphs and pools
  node representations for graph classification.
