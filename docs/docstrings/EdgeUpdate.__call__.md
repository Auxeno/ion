Update edge features from their incident nodes.

Parameters
----------
x : jax.Array["n i", float]
    Feature matrix for `n` nodes.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
x_edge : jax.Array["e f", float] | None, default=None
    Current feature vector for each directed edge. Keyword-only. When omitted,
    the edge model receives only sender and receiver features.

Returns
-------
jax.Array["e o", float]
    Updated features with one row per edge. The feature dimension comes from
    `edge_model`.
