Update edge and destination-node features.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array["s i", float], jax.Array["t j", float]]
    Node features, or `(x_src, x_dst)` for bipartite message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
x_edge : jax.Array["e f", float] | None, default=None
    Current feature vector for each directed edge. Keyword-only. When omitted,
    the edge model receives only sender and receiver features.
edge_mask : jax.Array["e", bool] | None, default=None
    Boolean edge mask where `True` keeps an updated edge in node aggregation and
    `False` excludes it. Masked edges are still updated and returned.

Returns
-------
tuple[jax.Array["t o", float], jax.Array["e m", float]]
    Updated destination-node and edge features. Their feature dimensions come
    from `node_model` and `edge_model`, respectively.
