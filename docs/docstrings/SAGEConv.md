GraphSAGE layer ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)).

Pools neighbour features with a permutation-invariant aggregate, then combines
them with a separately transformed copy of the central node:

\[
x'_i = W_n\,\operatorname{agg}_{j\in\mathcal N(i)}(x_j) + W_r x_i.
\]

Parameters
----------
in_dim : int | tuple[int, int]
    Input node feature dimension. Pass `(src_dim, dst_dim)` for bipartite
    source and destination features with different widths.
out_dim : int
    Output node feature dimension.
aggregate : str, default='mean'
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
    Neighbour transform of shape `(src_dim, out_dim)`.
w_root : Param | None
    Destination-root transform of shape `(dst_dim, out_dim)`. `None` when
    `use_root_weight=False`.
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

sage = gnn.SAGEConv(in_dim, out_dim, aggregate="max", key=key)
y = sage(x, senders, receivers)  # (3, 16) -> (3, 32)
```
