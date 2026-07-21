Graph Attention Network layer (Velickovic et al., 2018).

Learns attention weights over each node's neighborhood using LeakyReLU-gated additive attention: `e_ij = LeakyReLU(a_l^T W h_i + a_r^T W h_j)`. Multi-head attention is supported; heads are concatenated.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension. Must be divisible by `num_heads`; each head
    produces `out_dim // num_heads` features, concatenated to `out_dim`.
num_heads : int, default=1
    Number of attention heads.
edge_dim : int | None, default=None
    Per-edge feature dimension. When set, edge features are projected into the
    multi-head space and added to the attention logits before the LeakyReLU
    gate. When `None`, no edge parameters are created.
negative_slope : float, default=0.2
    Negative slope of the LeakyReLU used to gate attention logits.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Projection weight initializer. Glorot uniform by default (activation-
    agnostic), since the projection feeds a LeakyReLU attention mechanism.
att_init : Initializer
    Attention vector initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Projection weight of shape `(in_dim, out_dim)`.
att_sender, att_receiver : Param
    Per-head attention vectors of shape `(num_heads, out_dim // num_heads)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `bias=False`.
w_edge, att_edge : Param | None
    Edge projection and attention parameters. `None` unless `edge_dim` is set.

Notes
-----
`edge_dim` and `x_edge` must be used together; setting one without the other raises an error. `x_edge` and `edge_mask` are keyword-only. Pass a boolean `edge_mask` of shape `(e,)` to disable individual edges: masked edges get `-inf` attention logits (zero weight) and their edge features are zeroed. This is useful for padded batches or dropping edges at inference without rebuilding the edge index.

Examples
--------
>>> gat = gnn.GATConv(16, 32, num_heads=4, key=key)
>>> y = gat(x, senders, receivers)                         # (n, 16) -> (n, 32)
>>> gat = gnn.GATConv(16, 32, num_heads=4, edge_dim=8, key=key)
>>> y = gat(x, senders, receivers, x_edge=x_edge)          # x_edge: (e, 8)
