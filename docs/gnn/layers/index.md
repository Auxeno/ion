# Graph neural network layers

Message-passing layers receive a node feature matrix and parallel COO `senders`/`receivers` arrays. Graph readout layers instead receive `graph_ids` to pool nodes into graph representations. See the [GNN guide](../guide.md) for graph representation, message passing, self-loops, batching, and pooling.

## Choose a layer

| Family | Layers | Neighbour aggregation |
|---|---|---|
| [Edge Update](edge.md) | `EdgeUpdate` | Independent update from the incident nodes and current edge |
| [Gated Graph Convolution](gated_gcn.md) | `GatedGCNConv` | Normalized feature-wise edge gates |
| [Graph Convolution](gcn.md) | `GCNConv`, `GraphConv` | Normalized or optionally weighted sum |
| [Graph Attention](gat.md) | `GATConv`, `GATv2Conv`, `TransformerConv` | Learned attention weights |
| [Graph Isomorphism Network](gin.md) | `GINConv`, `GINEConv` | Sum plus a separate central-node term |
| [GraphSAGE](sage.md) | `SAGEConv` | Mean, max, or sum pooling plus a root term |
| [Graph Readout](readout.md) | `GlobalAttentionPool` | Learned attention-weighted sum across nodes |
