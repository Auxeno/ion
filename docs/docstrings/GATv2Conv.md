Dynamic Graph Attention Network layer ([Brody et al., 2022](https://arxiv.org/abs/2105.14491)).

Fixes a limitation of `GATConv` where attention rankings are "static" (identical for all query nodes). GATv2 applies the LeakyReLU *after* combining sender and receiver features, `e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)`, so attention scores depend on both nodes. The constructor interface matches `GATConv`.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension. Must be divisible by `num_heads`.
num_heads : int, default=1
    Number of attention heads.
edge_dim : int | None, default=None
    Per-edge feature dimension. When set, edge features are added *inside* the
    LeakyReLU (before the attention dot product), so the nonlinearity mixes node
    and edge information. When `None`, no edge parameters are created.
negative_slope : float, default=0.2
    Negative slope of the LeakyReLU used to combine features.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Projection weight initializer. Glorot uniform by default.
att_init : Initializer
    Attention vector initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_sender, w_receiver : Param
    Separate sender and receiver projections of shape `(in_dim, out_dim)`.
att : Param
    Per-head attention vector of shape `(num_heads, out_dim // num_heads)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `bias=False`.
w_edge : Param | None
    Edge projection. `None` unless `edge_dim` is set.

Notes
-----
Structural difference from `GATConv`: two weight matrices (`w_sender`, `w_receiver`) instead of one, and a single attention vector (`att`) instead of two, so attention is computed per-edge rather than decomposed to node-level scores. The `edge_dim` / `x_edge` / `edge_mask` interface is identical to `GATConv`.

Example
-------
```python
gat = gnn.GATv2Conv(16, 32, num_heads=4, key=key)
y = gat(x, senders, receivers)                         # (n, 16) -> (n, 32)

gat = gnn.GATv2Conv(16, 32, num_heads=4, edge_dim=8, key=key)
y = gat(x, senders, receivers, x_edge=x_edge)          # x_edge: (e, 8)
```
