Apply relational graph convolution to node features.

Parameters
----------
x : jax.Array["n i", float]
    Node feature matrix.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
edge_type : jax.Array["e", int]
    Edge-type index selecting the neighbour transform for each edge.
    Keyword-only.

Returns
-------
jax.Array["n o", float]
    Updated node features with `out_dim` features per node.

Note
----
Messages are averaged separately for each edge type and receiver, then summed
with the root projection. Self-loops are therefore unnecessary.
