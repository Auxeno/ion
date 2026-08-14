Apply scaled dot-product attention over graph edges.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array["s i", float], jax.Array["t j", float]]
    Node features, or `(x_src, x_dst)` for bipartite message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
x_edge : jax.Array["e f", float] | None, default=None
    Feature vector for each edge. Keyword-only. Required when `edge_dim` was
    set at construction and otherwise must be omitted.
edge_mask : jax.Array["e", bool] | None, default=None
    Boolean edge mask where `True` keeps an edge and `False` excludes it.
    Keyword-only.

Returns
-------
jax.Array["t o", float]
    Attended destination-node features with the heads concatenated into `out_dim`.

Note
----
With `use_root_weight=True`, a learned projection of each node's features is added
separately, so self-loops are unnecessary. `use_beta=True` combines that projection
and the aggregated messages with a sigmoid gate instead.
