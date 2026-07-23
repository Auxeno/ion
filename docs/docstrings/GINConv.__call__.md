Apply graph isomorphism convolution to node features.

Parameters
----------
x : jax.Array["n i", float]
    Feature matrix for `n` nodes, matching the input dimension of `mlp`.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.

Returns
-------
jax.Array["n o", float]
    Node features returned by `mlp` after sum aggregation.

Warning
-------
Do not add self-loops: a node's own features already enter through the `(1 + eps)` term, so adding them double-counts.