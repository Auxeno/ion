Jointly update node and edge features.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array["s i", float], jax.Array["t j", float]]
    Node features, or `(x_src, x_dst)` for bipartite message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
x_edge : jax.Array["e f", float]
    Feature matrix with one row per directed edge and `edge_dim` features per
    row. Keyword-only.

Returns
-------
tuple[jax.Array["t o", float], jax.Array["e o", float]]
    Updated destination-node and edge features, both with `out_dim` features per row.

Note
----
Self-loops are not needed because each node enters through `w_self`. A node
with no incoming edges receives only its root term and node bias.
