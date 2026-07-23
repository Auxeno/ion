Graph Convolutional Network layer ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)).

Applies a shared linear transform then aggregates over each node's neighborhood with symmetric degree normalization, `D^{-1/2} A D^{-1/2} X W`.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. Glorot uniform by default, matching `Linear`.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Weight matrix of shape `(in_dim, out_dim)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `bias=False`.

Example
-------
```python
# Three-node directed cycle, with self-loops
num_nodes, in_dim, out_dim = 3, 16, 32
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)

gcn = gnn.GCNConv(in_dim, out_dim, key=key)
y = gcn(x, senders, receivers)  # (3, 16) -> (3, 32)
```
