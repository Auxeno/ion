# GraphSAGE

Sample-and-aggregate graph convolution. Neighbour features are pooled by mean,
max, or sum, and the central node enters through a separate root weight.
Operates on graphs in COO format; see the [GNN guide](../guide.md) for the
shared array, self-loop, and batching conventions.

::: ion.gnn.SAGEConv
