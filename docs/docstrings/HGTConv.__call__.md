Apply heterogeneous graph attention to node features.

Parameters
----------
x : jax.Array["n i", float]
    Node feature matrix.
senders : jax.Array["e", int]
    Source node index for each directed edge.
receivers : jax.Array["e", int]
    Destination node index for each directed edge.
node_type : jax.Array["n", int]
    Node-type index selecting each node's projections. Keyword-only.
edge_type : jax.Array["e", int]
    Edge-type index selecting each edge's attention and message transforms.
    Keyword-only.
edge_mask : jax.Array["e", bool] | None, default=None
    Boolean edge mask where `True` keeps an edge in node aggregation and `False`
    excludes it. Keyword-only.

Returns
-------
jax.Array["n o", float]
    Updated node features with `out_dim` features per node.

Note
----
Attention is normalized over all incoming edges of each receiver, across edge
types. With `use_skip=True`, the result is gated with the input features.
