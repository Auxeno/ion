Pool node features into per-seed graph representations.

Parameters
----------
x : jax.Array["n i", float]
    Feature matrix for `n` nodes, with `in_dim` features per node.
graph_ids : jax.Array["n", int]
    Graph index for each node.
num_graphs : int
    Number of output graphs. Sets the leading output dimension and preserves
    zero rows for graphs with no nodes.

Returns
-------
jax.Array["g s o", float]
    One representation per graph and seed, with the heads concatenated into
    `out_dim`. Graphs with no nodes return the output bias alone.
