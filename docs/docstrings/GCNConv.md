Graph Convolutional Network layer (Kipf & Welling, 2017).

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
    Weight initializer. He normal by default, matching `Linear`, since GCNConv
    is typically followed by ReLU.
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

Notes
-----
No activation is included; compose with `jax.nn.relu` between layers. Self-loops are not added automatically: call `gnn.add_self_loops` first so each node includes its own features in the aggregation.

Examples
--------
>>> gcn = gnn.GCNConv(16, 32, key=key)
>>> y = gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
