Pool node features into graph representations.

Parameters
----------
x : jax.Array["n i", float]
    Feature matrix for `n` nodes.
graph_ids : jax.Array["n", int]
    Graph index for each node.
num_graphs : int
    Number of output graphs. Sets the leading output dimension and preserves
    zero rows for graphs with no nodes.

Returns
-------
jax.Array["g o", float]
    Attention-weighted graph representations. The feature dimension comes from
    `value`, or from `x` when no value module is supplied.
