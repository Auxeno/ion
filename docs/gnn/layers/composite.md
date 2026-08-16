# Composite

Compose caller-supplied models into the edge and node updates of a Graph Network block. `EdgeUpdate` transforms every edge, `NodeUpdate` reduces edges selected by an optional `edge_mask` into nodes, and `GraphNetwork` performs both stages while returning every updated edge. `EdgeUpdate` takes no mask, because it updates edges without aggregating them.

::: ion.gnn.GraphNetwork

::: ion.gnn.EdgeUpdate

::: ion.gnn.NodeUpdate
