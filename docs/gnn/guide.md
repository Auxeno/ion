# Graph neural networks

For graph layers and their APIs, see the [GNN layer reference](layers/index.md).

This guide represents a graph with JAX arrays, follows messages through it, and builds node-level and graph-level predictors.

- [COO format](#coo-format)
- [Message passing](#message-passing)
- [Aggregating messages](#aggregating-messages)
- [Custom message passing](#custom-message-passing)
- [Self-loops](#self-loops)
- [Bipartite graphs](#bipartite-graphs)
- [Feature propagation](#feature-propagation)
- [Building a GNN](#building-a-gnn)
- [Batching graphs](#batching-graphs)
- [Graph pooling](#graph-pooling)
- [Static shapes](#static-shapes)
- [Further examples](#further-examples)

## COO format

A graph starts with an ordinary array of feature vectors. Each row of `x` represents one node. The example has six nodes with exclusive one-hot features, so it is always possible to tell where a contribution originated:

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

The rows are nodes and the columns are features, so `x` has shape `(num_nodes, node_dim)`, here `(6, 6)`. An ordinary neural network layer can transform these rows independently. A graph layer additionally uses edges to exchange information between connected rows.

Edges are stored in two parallel one-dimensional arrays:

```python
senders   = jnp.array([0, 1, 0, 2, 1, 3, 1, 4, 2, 4, 2, 5, 3, 4, 4, 5])
receivers = jnp.array([1, 0, 2, 0, 3, 1, 4, 1, 4, 2, 5, 2, 4, 3, 5, 4])
```

The arrays have shape `(num_edges,)`. Read matching entries at the same index as one directed edge:

| `senders[i]` | `receivers[i]` | Meaning |
|---:|---:|---|
| **0** | **1** | **Node 0 sends to node 1** |
| 1 | 0 | Node 1 sends to node 0 |
| 0 | 2 | Node 0 sends to node 2 |
| 2 | 0 | Node 2 sends to node 0 |
| 1 | 3 | Node 1 sends to node 3 |
| 3 | 1 | Node 3 sends to node 1 |

This is coordinate, or COO, sparse format. The first two entries encode the undirected connection `0 <-> 1` as two directed edges:

```python
senders[0], receivers[0]  # 0, 1: node 0 sends to node 1
senders[1], receivers[1]  # 1, 0: node 1 sends to node 0
```

The complete arrays above produce this graph. The highlighted arrow is edge 0.

<iframe
  class="gnn-plot gnn-plot--graph"
  src="../../assets/gnn-coo-graph.html"
  title="Six-node graph built from the COO edge arrays"
  loading="eager"
></iframe>

Edges in COO are always directed. If the relationship should be undirected, include both directions as above.

### Optional edge data

Many graphs only need node features and connectivity. Some applications also attach information to each edge. Molecular graphs are a natural example because bonds can have different types.

Acetic acid, `CH3COOH`, contains carbon, hydrogen, and oxygen atoms connected by single and double bonds. The first plot shows its chemical structure. The second replaces every atom and bond with a one-hot feature vector.

<iframe
  class="gnn-plot gnn-plot--molecule"
  src="../../assets/gnn-acetic-acid.html"
  title="Acetic acid and its one-hot graph representation"
  loading="eager"
></iframe>

Use feature order carbon, hydrogen, oxygen for the atoms. The atom rows follow the conventional `CH3COOH` formula order:

```python
x_molecule = jnp.array([
    [1.0, 0.0, 0.0],  # C
    [0.0, 1.0, 0.0],  # H
    [0.0, 1.0, 0.0],  # H
    [0.0, 1.0, 0.0],  # H
    [1.0, 0.0, 0.0],  # C
    [0.0, 0.0, 1.0],  # O
    [0.0, 0.0, 1.0],  # O
    [0.0, 1.0, 0.0],  # H
], dtype=jnp.float32)
```

As before, each undirected connection becomes two directed edges. The edge features use order single bond, double bond:

```python
senders_molecule   = jnp.array([0, 1, 0, 2, 0, 3, 0, 4, 4, 5, 4, 6, 6, 7])
receivers_molecule = jnp.array([1, 0, 2, 0, 3, 0, 4, 0, 5, 4, 6, 4, 7, 6])

x_edge_molecule = jnp.array([
    [1.0, 0.0], [1.0, 0.0],  # C-H directed edge pair
    [1.0, 0.0], [1.0, 0.0],  # C-H
    [1.0, 0.0], [1.0, 0.0],  # C-H
    [1.0, 0.0], [1.0, 0.0],  # C-C
    [0.0, 1.0], [0.0, 1.0],  # C=O
    [1.0, 0.0], [1.0, 0.0],  # C-O
    [1.0, 0.0], [1.0, 0.0],  # O-H
], dtype=jnp.float32)
```

`x_edge_molecule[i]` describes the same directed edge as `senders_molecule[i]` and `receivers_molecule[i]`. `GATConv`, `GATv2Conv`, and `TransformerConv` accept these feature vectors. Construct the layer with the matching `edge_dim`:

```python
key = jax.random.key(0)
gat = gnn.GATConv(in_dim=3, out_dim=16, edge_dim=2, key=key)
h = gat(
    x_molecule,
    senders_molecule,
    receivers_molecule,
    x_edge=x_edge_molecule,
)
```

`GINEConv` also takes edge features, but adds them straight to the sender features instead of projecting them, so `x_edge` must already be at the node dimension and there is no `edge_dim` to set.

The attention layers also accept one boolean mask value per directed edge. `True` keeps an edge and `False` makes the layer ignore it:

```python
edge_mask = jnp.ones(senders_molecule.shape, dtype=bool)
edge_mask = edge_mask.at[jnp.array([8, 9])].set(False)  # ignore C=O both ways
h = gat(
    x_molecule,
    senders_molecule,
    receivers_molecule,
    x_edge=x_edge_molecule,
    edge_mask=edge_mask,
)
```

For an undirected relationship such as a bond, mask both directed edges to exclude it completely. `GraphConv` does not accept feature vectors or a mask, but it can scale each message with a scalar `edge_weight`.

### Updating edge features

`EdgeUpdate` learns a new representation for each edge from its sender, receiver, and current edge features. Its caller-supplied model receives their concatenation:

```python
edge_model = nn.MLP([2 * node_dim + edge_dim, hidden_dim, out_dim], key=key)
update = gnn.EdgeUpdate(edge_model)
x_edge = update(x, senders, receivers, x_edge=x_edge)
```

`GatedGCNConv` updates nodes and edges together. The updated edges become feature-wise gates on the node messages, and both outputs share `out_dim`:

```python
conv = gnn.GatedGCNConv(node_dim, out_dim, edge_dim=edge_dim, key=key)
x, x_edge = conv(x, senders, receivers, x_edge=x_edge)
x, x_edge = jax.nn.relu(x), jax.nn.relu(x_edge)
```

Like Ion's other convolutions, neither layer applies an activation or normalization. Add those explicitly when building a block.

The complete graph representation consists of JAX arrays:

| Array | Shape | Meaning |
|---|---|---|
| `x` | `(num_nodes, node_dim)` | Feature vector for each node |
| `senders` | `(num_edges,)` | Source node of each directed edge |
| `receivers` | `(num_edges,)` | Destination node of each directed edge |
| `edge_weight` | `(num_edges,)` | Optional scalar weight for each edge |
| `x_edge` | `(num_edges, edge_dim)` | Optional feature vector for each edge |
| `edge_mask` | `(num_edges,)` | Optional boolean selecting active edges |

### Dense adjacency

Datasets and spectral methods often use an `(n, n)` adjacency matrix instead. Convert in either direction:

```python
adjacency = gnn.to_adjacency(senders, receivers, num_nodes)
senders, receivers = gnn.from_adjacency(adjacency)
```

`from_adjacency` produces a data-dependent number of edges, so pass `num_edges` to call it under `jax.jit`. Spare slots hold the out-of-range index `num_nodes`, which segment reductions drop.

## Message passing

A graph layer uses each directed edge to route information:

1. The sender produces a message.
2. The edge optionally scales or transforms that message.
3. Messages with the same receiver are aggregated.

`senders` has one entry per edge. Indexing `x` with it copies the feature row of each edge's source node:

```python
messages = x[senders]
messages.shape  # (16 edges, 6 features)

messages[0]  # x[senders[0]] = x[0]
messages[1]  # x[senders[1]] = x[1]
```

Nothing has been combined yet. `messages[i]` is the feature vector travelling along edge `i`, from `senders[i]` to `receivers[i]`.

## Aggregating messages

`segment_sum` groups message rows by their receiver. Message `i` is added to output row `receivers[i]`:

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

The four messages are grouped into row 4 because their receiver is 4. Node 4's own feature is absent because the graph does not yet contain a `4 -> 4` self-loop. This send-then-group operation is the basis of the GCN, GraphConv, graph attention, GIN, and GraphSAGE layers.

## Custom message passing

`GraphNetwork` composes caller-supplied edge and node models around the same
reduction. The edge model sees each sender, receiver, and current edge; its
updated edges are aggregated and passed to the node model with the current node:

```python
key_edge, key_node = jax.random.split(key)
edge_model = nn.MLP(
    [2 * node_dim + edge_dim, hidden_dim, message_dim],
    key=key_edge,
)
node_model = nn.MLP(
    [node_dim + message_dim, hidden_dim, out_dim],
    key=key_node,
)

network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
x, x_edge = network(x, senders, receivers, x_edge=x_edge)
```

The updated edges are both returned and used as messages. Omit `x_edge` to
construct them from the two incident nodes alone, or pass another compatible
reduction such as `aggregate=gnn.segment_mean`. The supplied models own all
parameters; `GraphNetwork` applies no activation, normalization, or residual.

The two stages are also available separately. `EdgeUpdate` creates new edge
features from their incident nodes. `NodeUpdate` accepts already-computed edge
features, so node-only message passing does not need an edge model:

```python
x_edge = x[senders]
node_model = nn.Linear(node_dim + node_dim, out_dim, key=key)
update = gnn.NodeUpdate(node_model, aggregate=gnn.segment_mean)
x = update(x, senders, receivers, x_edge=x_edge)
```

Both index arrays remain in the call signature because each feature row belongs
to the directed edge `(senders[i], receivers[i])`; only `receivers` is needed
for the reduction itself.

## Self-loops

A node is not automatically its own neighbour. `GCNConv`, `GATConv`, and `GATv2Conv` normally need self-loop edges so each node can retain its current features:

```python
num_nodes = x.shape[0]
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)
```

This appends `(0, 0)`, `(1, 1)`, through `(5, 5)`. Self-loops are explicit and are never added inside a layer.

| Layer | Add self-loops? | Reason |
|---|---|---|
| `GCNConv` | Normally yes | Include the node's current features in aggregation |
| `GraphConv` | No | The separate root weight already includes the node |
| `SAGEConv` | No | The root weight already includes the node |
| `GATConv`, `GATv2Conv` | Normally yes | Let a node attend to itself |
| `TransformerConv` | No | The root weight includes the node by default |
| `GINConv`, `GINEConv` | No | The `(1 + eps)` term already includes the node |
| `GraphNetwork` | No | The node model receives the current node directly |
| `NodeUpdate` | No | The node model receives the current node directly |
| `GatedGCNConv` | No | The root weight already includes the node |

`add_self_loops` appends one loop for every node. It does not check whether the input already contains self-loops, so avoid calling it twice on the same edge arrays. When using edge features or a mask, also append one corresponding row or value for every new self-loop.

`remove_self_loops` is the inverse and drops every `i -> i` edge, which is useful when a dataset ships with self-loops already present and you want to control them yourself:

```python
senders, receivers = gnn.remove_self_loops(senders, receivers)
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)
```

How many edges it removes depends on the data, so the output shape is not known ahead of time and the call cannot go inside `jax.jit`. Treat it as a data preparation step. Edge features and masks are not filtered for you; apply the same `senders != receivers` mask to them.

## Bipartite graphs

Most message-passing layers can also send between two distinct node sets. Pass
the node features as `(x_src, x_dst)` in this case. `senders` indexes rows of
`x_src`, `receivers` indexes rows of `x_dst`, and the layer returns one row for
each destination node:

```python
x_src = jnp.ones((num_src, src_dim))
x_dst = jnp.ones((num_dst, dst_dim))

conv = gnn.GraphConv((src_dim, dst_dim), out_dim, key=key)
y_dst = conv((x_src, x_dst), senders, receivers)

y_dst.shape  # (num_dst, out_dim)
```

For layers with learned input projections, `in_dim` may be an integer or a
`(src_dim, dst_dim)` tuple. An integer uses the same feature width for both node
sets. `GINConv` and `GINEConv` instead receive their update network from the
caller; because their update equations add source and destination features,
the two node sets must have the same feature width. `GINEConv` edge features
must share that width too.

`GraphConv`, `SAGEConv`, `GATConv`, `GATv2Conv`, `TransformerConv`, `GINConv`,
`GINEConv`, `GraphNetwork`, `EdgeUpdate`, `NodeUpdate`, and `GatedGCNConv`
accept bipartite inputs. `GCNConv` does not: its symmetric degree normalization
assumes one shared node domain.

Passing a single array retains the usual homogeneous behavior and is equivalent
to passing `(x, x)`. A bipartite call only updates the destination node set. To
update both partitions, apply a second layer in the reverse direction with
reversed edge indices. Self-loops do not apply: the two node sets are distinct,
so no edge joins a node to itself.

## Line graphs

`EdgeUpdate` and `GatedGCNConv` update an edge from its two incident nodes. To let neighbouring edges communicate directly, instead convert the graph so that its edges are the nodes:

```python
senders, receivers = gnn.remove_self_loops(senders, receivers)
senders, receivers, kept = gnn.to_undirected(senders, receivers)
x_edge = jnp.concatenate([x_edge, x_edge])[kept]

line_senders, line_receivers, shared = gnn.line_graph(senders, receivers)
x_edge = conv(x_edge, line_senders, line_receivers)
```

Two edges are joined when one ends where the other begins, so they form a two-hop path through a shared node. A line-graph node is a row of the edge arrays, which is why `x_edge` becomes the node features with no rearranging, and why a convolution that only knows how to update nodes ends up updating edges.

The third return value is the node each pair of edges passes through. It is the natural place for line-graph edge features, either `x[shared]` or the angle between the two edges when nodes carry positions.

An undirected graph stored with both directions gives each direction its own line-graph node, so `(i, j)` and `(j, i)` hold independent state. `non_backtracking` defaults to `True` so an edge never sends a message straight back down its own reverse; pass `False` when the graph is genuinely directed and `i -> v -> i` is a real path. The result has `sum(indeg(v) * outdeg(v))` edges, which grows with the square of the degree, so a graph with high-degree hubs produces a very large line graph.

## Feature propagation

A GCN applies a shared linear transformation and then combines features using degree-normalized edges. To make the graph operation visible by itself, the plot below fixes the linear transformation to the identity and omits the bias and activation. It repeatedly applies only the normalized aggregation:

\[
D^{-1/2} A D^{-1/2} x
\]

The input is the one-hot matrix constructed above, with self-loops added. Choose a step from 0 to 5. Click any node to follow its original feature, and hover any node to inspect its complete mixed feature vector.

<iframe
  class="gnn-plot gnn-plot--propagation"
  src="../../assets/gnn-feature-propagation.html"
  title="One-hot node features mixing over five message-passing steps"
  loading="eager"
></iframe>

One step lets a feature reach immediate neighbours. By two steps, feature `e_0` has reached the entire example graph. Later steps continue mixing the values even though the set of reachable nodes no longer changes.

These are five message-passing steps, which correspond to five GCN layers when their learned transformations are replaced by the identity. Calling the same layer five separate times on the original `x` would instead repeat the same one-step calculation.

## Building a GNN

The previous plot fixed the feature transformation to the identity so only the graph operation was visible. `GCNConv` restores that transformation and makes it learnable: each layer multiplies the node features by a weight matrix, then applies the same normalized aggregation as the plot. Training adjusts the weight matrix, so the layer learns which projection of the input features to mix across edges:

```python
key = jax.random.key(0)
gcn = gnn.GCNConv(in_dim=6, out_dim=16, key=key)

# (6 nodes, 6 input features) -> (6 nodes, 16 output features)
h = gcn(x, senders, receivers)
```

Each layer performs one message-passing step. Stacking layers lets information travel across progressively longer paths:

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

The linear classifier acts on each node independently after the graph layers have mixed information between connected nodes. Ion modules are JAX [pytrees](https://docs.jax.dev/en/latest/pytrees.html), so `jax.jit`, `jax.grad`, and the usual Ion optimizer workflow apply without graph-specific transforms.

## Batching graphs

Ordinary neural network inputs are often stacked along a leading batch dimension:

```python
x.shape  # (batch, items, features)
```

Graphs frequently contain different numbers of nodes and edges, so they cannot be stacked this way without padding. Ion instead concatenates them into one disconnected graph:

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

`batch_graphs` offsets the edge indices of each graph and records which graph each node belongs to in `graph_ids`. There are no edges between the packed graphs, so message passing cannot mix information between examples. Applying a graph layer to the disconnected graph is equivalent to applying it to each graph separately:

```python
h = gcn(x, senders, receivers)
```

## Graph pooling

Node-level tasks use one output row per node. Graph-level tasks reduce node representations into one row per graph:

```python
graph_h = gnn.mean_pool(h, graph_ids, num_graphs=2)
logits = graph_classifier(graph_h)

graph_h.shape  # (2, hidden_dim)
logits.shape   # (2, num_classes)
```

`graph_ids` serves the role normally played by a batch index, allowing the packed node representations to be reduced back to one row per graph.

Ion provides three fixed graph readouts:

| Operation | Reduction |
|---|---|
| `mean_pool` | Mean node representation |
| `sum_pool` | Sum node representation |
| `max_pool` | Maximum node representation |

Sum pooling preserves graph-size information, while mean pooling normalizes it away. The [Operations reference](operations.md) documents the complete pooling APIs.

`GlobalAttentionPool` learns which nodes matter. Its `score` module produces one importance logit per node; softmax normalizes those logits within each graph. Its optional `value` module produces the features included in the weighted sum:

```python
key_score, key_value = jax.random.split(key)
attention_pool = gnn.GlobalAttentionPool(
    score=nn.Linear(hidden_dim, 1, use_bias=False, key=key_score),
    value=nn.Linear(hidden_dim, hidden_dim, key=key_value),
)
graph_h = attention_pool(h, graph_ids, num_graphs=2)
```

The score and value modules supply all of the readout's parameters. Omit `value` to pool `h` directly. A bias on a single linear score is redundant because softmax is unchanged by a constant shift. See the [Pooling reference](layers/pool.md) for the full API.

## Static shapes

Call `batch_graphs` outside `jax.jit`. It accepts Python sequences of differently shaped arrays and constructs the packed arrays that enter the compiled model. `unbatch_graphs` inverts it, splitting per-node outputs back into one array per graph, and is likewise a host-side call.

JAX compiles a function for the shapes it receives. If the packed number of nodes or edges changes, the compiled function may be retraced for the new shapes. Pad nodes and edges to fixed maximum sizes when a workload requires static shapes.

### Shape labels

| Label | Meaning |
|---|---|
| `n` | Number of nodes |
| `e` | Number of directed edges |
| `g` | Number of graphs |
| `l` | Number of directed edges in a line graph |
| `d` | General node feature dimension |
| `i`, `o` | Input and output feature dimensions |
| `h`, `k` | Attention heads and per-head dimension |
| `f` | Edge feature dimension |

Each layer page defines how these labels apply to its inputs, parameters, and outputs.

## Further examples

- [GNN on Cora](../examples/gnn-cora.md) performs node classification on one citation graph.
- [GNN on BBBP](../examples/gnn-bbbp.ipynb) batches molecular graphs and pools node representations for graph classification.
