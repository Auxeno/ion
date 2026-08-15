Apply graph convolution to node features.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array["s i", float], jax.Array["t j", float]]
    Node features, or `(x_src, x_dst)` for bipartite message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
edge_weight : jax.Array["e", float] | None, default=None
    Optional scalar weight for each directed edge. Keyword-only.

Returns
-------
jax.Array["t o", float]
    Aggregated destination-node features with `out_dim` features per node.

Note
----
Self-loops are not needed because the central node enters through `w_root`.
If included, a self-loop contributes through both the neighbour and root
paths. A node with no incoming edges receives only its root term and bias.
