# Relational

Layers for typed graphs, where a per-edge `edge_type` array selects which transform each message goes through. `RGCNConv` gives every relation its own neighbour transform, with optional basis sharing between them. `HGTConv` adds a per-node `node_type` array and attends with type-dependent projections.

::: ion.gnn.RGCNConv

::: ion.gnn.HGTConv
