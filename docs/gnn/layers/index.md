# Graph neural network layers

Every graph layer receives a node feature matrix and parallel COO
`senders`/`receivers` arrays. See the [GNN guide](../guide.md) for graph
representation, message passing, self-loops, batching, and pooling.

## Choose a layer

| Family | Layers | Neighbour aggregation |
|---|---|---|
| [GCN](gcn.md) | `GCNConv` | Fixed symmetric degree normalization |
| [GAT](gat.md) | `GATConv`, `GATv2Conv` | Learned attention weights |
| [GIN](gin.md) | `GINConv` | Sum plus a separate central-node term |
