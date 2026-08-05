Graph convolutional layer ([Morris et al., 2019](https://arxiv.org/abs/1810.02244)).

Sum-aggregates neighbour features and combines them with a separately
transformed copy of the central node,
\(W_n \sum_{j \in \mathcal{N}(i)} e_{ji} x_j + W_s x_i\). When omitted,
each scalar edge weight \(e_{ji}\) is one.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension.
use_bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer for both transforms. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_neigh : Param
    Neighbour transform of shape `(in_dim, out_dim)`.
w_self : Param
    Root transform of shape `(in_dim, out_dim)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `use_bias=False`.

Example
-------
```python
# Three-node directed cycle; self-loops are not needed
num_nodes, in_dim, out_dim = 3, 16, 32
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
edge_weight = jnp.array([0.5, 1.0, 2.0])

conv = gnn.GraphConv(in_dim, out_dim, key=key)
y = conv(x, senders, receivers, edge_weight=edge_weight)
```

Note
----
Without edge weights, this update matches `SAGEConv` with `aggregator="sum"`,
`normalize=False`, and `use_root_weight=True`. `GraphConv` provides the narrower
canonical operator and supports scalar edge weights.
