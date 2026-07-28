# Graph neural network layers

Every graph layer receives a node feature matrix and parallel COO
`senders`/`receivers` arrays. See the [GNN guide](../guide.md) for graph
representation, message passing, self-loops, batching, and pooling.

## Choose a layer

| Family | Layers | Neighbour aggregation |
|---|---|---|
| [Graph Convolution](gcn.md) | `GCNConv`, `GraphConv` | Normalized or optionally weighted sum |
| [Graph Attention](gat.md) | `GATConv`, `GATv2Conv`, `TransformerConv` | Learned attention weights |
| [Graph Isomorphism Network](gin.md) | `GINConv` | Sum plus a separate central-node term |
| [GraphSAGE](sage.md) | `SAGEConv` | Mean, max, or sum pooling plus a root term |
