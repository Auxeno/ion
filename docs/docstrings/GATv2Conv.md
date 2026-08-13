Dynamic Graph Attention Network layer ([Brody et al., 2022](https://arxiv.org/abs/2105.14491)).

Fixes a limitation of `GATConv` where attention rankings are "static"
(identical for all query nodes). For each head, GATv2 applies the LeakyReLU
*after* combining sender and receiver features:

\[
\begin{gathered}
e_{ij}=a^\top\operatorname{LeakyReLU}(W_s h_i+W_r h_j),
\qquad
\alpha_{ij}=\operatorname{softmax}_{i\in\mathcal N(j)}(e_{ij}),
\\[4pt]
h'_j=\operatorname{Concat}_{k=1}^{H}\!\left(
\sum_{i\in\mathcal N(j)}\alpha_{ij}^{(k)}W_s^{(k)}h_i\right).
\end{gathered}
\]

This makes attention scores depend on both nodes. Optional edge features are
added inside the LeakyReLU, and the constructor interface matches `GATConv`.

Parameters
----------
in_dim : int | tuple[int, int]
    Input node feature dimension. Pass `(src_dim, dst_dim)` for bipartite
    source and destination features with different widths.
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
use_bias : bool, default=True
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
    Separate sender and receiver projections of shape `(src_dim, out_dim)` and
    `(dst_dim, out_dim)` respectively.
att : Param
    Per-head attention vector of shape `(num_heads, out_dim // num_heads)`.
b_out : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `use_bias=False`.
w_edge : Param | None
    Edge projection. `None` unless `edge_dim` is set.

Example
-------
```python
num_nodes, num_edges = 3, 3
in_dim, out_dim, edge_dim = 16, 32, 8
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

gat = gnn.GATv2Conv(in_dim, out_dim, num_heads=4, key=key)
y = gat(x, senders, receivers)  # (3, 16) -> (3, 32)

x_edge = jnp.ones((num_edges, edge_dim))
gat_edges = gnn.GATv2Conv(in_dim, out_dim, num_heads=4, edge_dim=edge_dim, key=key)
y = gat_edges(x, senders, receivers, x_edge=x_edge)  # (3, 16) -> (3, 32)
```

```python
gat = gnn.GATv2Conv((src_dim, dst_dim), out_dim, num_heads=4, key=key)
y_dst = gat((x_src, x_dst), senders, receivers)
```

Info
-----
Structural difference from `GATConv`: two weight matrices (`w_sender`, `w_receiver`) instead of one, and a single attention vector (`att`) instead of two, so attention is computed per-edge rather than decomposed to node-level scores. The `edge_dim` / `x_edge` / `edge_mask` interface is identical to `GATConv`.
