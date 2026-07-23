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

Info
----
No activation is included; compose with `jax.nn.relu` between layers.

Warning
-------
Self-loops are not added automatically. Call `gnn.add_self_loops` first, otherwise each node excludes its own features from the aggregation.

Example
-------
```python
gcn = gnn.GCNConv(16, 32, key=key)
y = gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
```
