GraphSAGE layer ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)).

Pools neighbour features with a permutation-invariant aggregator, then combines them with a separately transformed copy of the central node, \(W_n \, \mathrm{agg}(x_j) + W_s \, x_i\).

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension.
aggregator : str, default='mean'
    Neighbourhood pooling: `"mean"`, `"max"`, or `"sum"`.
normalize : bool, default=False
    Whether to L2 normalize each output node embedding.
use_root_weight : bool, default=True
    Whether to add the central node's own features through a separate weight.
use_bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_neigh : Param
    Neighbour transform of shape `(in_dim, out_dim)`.
w_self : Param | None
    Root transform of shape `(in_dim, out_dim)`. `None` when `use_root_weight=False`.
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

sage = gnn.SAGEConv(in_dim, out_dim, aggregator="max", key=key)
y = sage(x, senders, receivers)  # (3, 16) -> (3, 32)
```
