# Graph neural network layers

Message-passing layers receive a node feature matrix and parallel COO `senders`/`receivers` arrays. Except for `GCNConv`, they also accept an `(x_src, x_dst)` tuple for bipartite message passing. Graph pooling layers instead receive `graph_ids` to pool nodes into graph representations. See the [GNN guide](../guide.md) for graph representation, message passing, self-loops, bipartite inputs, batching, and pooling.

## Choose a layer

| Family | Layers |
|---|---|
| [Convolution](conv.md) | `GCNConv`, `GraphConv`, `SAGEConv` |
| [Attention](attention.md) | `GATConv`, `GATv2Conv`, `TransformerConv` |
| [Isomorphism](isomorphism.md) | `GINConv`, `GINEConv` |
| [Composite](composite.md) | `GraphNetwork`, `EdgeUpdate`, `NodeUpdate` |
| [Relational](relational.md) | `RGCNConv`, `HGTConv` |
| [Gated](gated.md) | `GatedGCNConv` |
| [Pooling](pool.md) | `GlobalAttentionPool`, `MultiHeadAttentionPool` |

## Feature support

A check means the layer accepts or produces that kind of feature array. Whether
an input is optional or required is described on the layer's reference page.

| Layer | Nodes<br><small>in&ensp;out</small> | Edges<br><small>in&ensp;out</small> | Bipartite |
|---|:---:|:---:|:---:|
| [`GCNConv`](conv.md#ion.gnn.GCNConv) | ✓&ensp;✓ | —&ensp;— | |
| [`GraphConv`](conv.md#ion.gnn.GraphConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`SAGEConv`](conv.md#ion.gnn.SAGEConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`GATConv`](attention.md#ion.gnn.GATConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`GATv2Conv`](attention.md#ion.gnn.GATv2Conv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`TransformerConv`](attention.md#ion.gnn.TransformerConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`GINConv`](isomorphism.md#ion.gnn.GINConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`GINEConv`](isomorphism.md#ion.gnn.GINEConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`GraphNetwork`](composite.md#ion.gnn.GraphNetwork) | ✓&ensp;✓ | ✓&ensp;✓ | ✓ |
| [`EdgeUpdate`](composite.md#ion.gnn.EdgeUpdate) | ✓&ensp;— | ✓&ensp;✓ | ✓ |
| [`NodeUpdate`](composite.md#ion.gnn.NodeUpdate) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`RGCNConv`](relational.md#ion.gnn.RGCNConv) | ✓&ensp;✓ | —&ensp;— | |
| [`HGTConv`](relational.md#ion.gnn.HGTConv) | ✓&ensp;✓ | —&ensp;— | |
| [`GatedGCNConv`](gated.md#ion.gnn.GatedGCNConv) | ✓&ensp;✓ | ✓&ensp;✓ | ✓ |

`GraphConv` can additionally scale messages with scalar `edge_weight` values and
`RGCNConv` and `HGTConv` select transforms with integer `node_type` and
`edge_type` values; none of these are edge feature vectors.

[`GlobalAttentionPool`](pool.md#ion.gnn.GlobalAttentionPool) and [`MultiHeadAttentionPool`](pool.md#ion.gnn.MultiHeadAttentionPool) are not message-passing layers: they take `graph_ids` rather than an edge list and return one row per graph, or one row per seed for the latter. The [graph operations](../operations.md#pooling) page covers the fixed mean, sum, and max readouts.
