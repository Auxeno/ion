Apply the GraphSAGE layer to node features.

Parameters
----------
x : jax.Array["n i", float]
    Feature matrix for `n` nodes, with `in_dim` features per node.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.

Returns
-------
jax.Array["n o", float]
    Aggregated node features with `out_dim` features per node.

Note
-------
Self-loops are not needed: the central node enters through the root weight.
A node with no incoming edges contributes only its own root term and bias.
