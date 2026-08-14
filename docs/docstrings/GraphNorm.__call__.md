Normalize node features independently within each graph.

Parameters
----------
x : jax.Array["n d", float]
    Feature matrix for `n` nodes.
graph_ids : jax.Array["n", int] | None, default=None
    Graph index for each node. When omitted, all nodes belong to one graph.
num_graphs : int | None, default=None
    Number of graphs in the packed batch. Required when `graph_ids` is given.

Returns
-------
jax.Array["n d", float]
    Normalized node features, with the same shape and dtype as `x`.
