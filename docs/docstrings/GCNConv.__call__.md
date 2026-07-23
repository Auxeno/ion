Apply graph convolution to node features.

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

Info
----
The graph is unbatched and uses COO edge arrays.

Warning
-------
Self-loops are not added automatically. Call `gnn.add_self_loops` first,
otherwise each node excludes its own features from the aggregation.
