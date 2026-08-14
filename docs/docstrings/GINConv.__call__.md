Apply graph isomorphism convolution to node features.

Parameters
----------
x : jax.Array["n i", float] | tuple[jax.Array["s i", float], jax.Array["t i", float]]
    Node features, or equal-width `(x_src, x_dst)` features for bipartite
    message passing.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.

Returns
-------
jax.Array["t o", float]
    Destination-node features returned by `mlp` after sum aggregation.

Note
-------
Do not add self-loops: a node's own features already enter through the `(1 + eps)` term, so adding them double-counts.
