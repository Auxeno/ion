# Graph neural network layers

Message-passing layers receive a node feature matrix and parallel COO `senders`/`receivers` arrays. Except for `GCNConv`, they also accept an `(x_src, x_dst)` tuple for bipartite message passing. Graph readout layers instead receive `graph_ids` to pool nodes into graph representations. See the [GNN guide](../guide.md) for graph representation, message passing, self-loops, bipartite inputs, batching, and pooling.

## Choose a layer

A check means the layer accepts or produces that kind of feature array. Whether
an input is optional or required is described on the layer's reference page.

| Layer | Nodes<br><small>in&ensp;out</small> | Edges<br><small>in&ensp;out</small> | Bipartite |
|---|:---:|:---:|:---:|
| [`GCNConv`](gcn.md#ion.gnn.GCNConv) | ✓&ensp;✓ | —&ensp;— | |
| [`GraphConv`](gcn.md#ion.gnn.GraphConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`SAGEConv`](sage.md#ion.gnn.SAGEConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`GATConv`](gat.md#ion.gnn.GATConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`GATv2Conv`](gat.md#ion.gnn.GATv2Conv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`TransformerConv`](gat.md#ion.gnn.TransformerConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`GINConv`](gin.md#ion.gnn.GINConv) | ✓&ensp;✓ | —&ensp;— | ✓ |
| [`GINEConv`](gin.md#ion.gnn.GINEConv) | ✓&ensp;✓ | ✓&ensp;— | ✓ |
| [`RGCNConv`](rgcn.md#ion.gnn.RGCNConv) | ✓&ensp;✓ | —&ensp;— | |
| [`GatedGCNConv`](gated_gcn.md#ion.gnn.GatedGCNConv) | ✓&ensp;✓ | ✓&ensp;✓ | ✓ |
| [`EdgeUpdate`](edge.md#ion.gnn.EdgeUpdate) | ✓&ensp;— | ✓&ensp;✓ | ✓ |

`GraphConv` can additionally scale messages with scalar `edge_weight` values and
`RGCNConv` selects a transform with integer `edge_type` values; neither is an
edge feature vector.
