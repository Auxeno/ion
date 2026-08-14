Update destination nodes from edge features.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array, jax.Array]
    Node features, or `(x_src, x_dst)` for bipartite message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge. Included to keep the graph-layer
    call signature consistent; edge features are already aligned with these edges.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
x_edge : jax.Array["e f", float]
    Feature vector for each directed edge. Keyword-only.
edge_mask : jax.Array["e", bool] | None, default=None
    Boolean edge mask where `True` keeps an edge in node aggregation and `False`
    excludes it.

Returns
-------
jax.Array["n o", float]
    Updated destination-node features. The feature dimension comes from
    `node_model`.
