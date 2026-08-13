# Convolution

Neighbourhood aggregation with learned feature transforms. `GCNConv` uses symmetric degree normalization; `GraphConv` uses an optionally edge-weighted sum with a separate root transform; `SAGEConv` pools neighbours with mean, max, or sum before its own root transform.

::: ion.gnn.GCNConv

::: ion.gnn.GraphConv

::: ion.gnn.SAGEConv
