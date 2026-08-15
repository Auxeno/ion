Graph Attention Network layer ([Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)).

Learns attention weights over each node's neighbourhood using LeakyReLU-gated
additive attention. For each head,

\[
\begin{gathered}
e_{ij}=\operatorname{LeakyReLU}(a_s^\top W h_i+a_r^\top W h_j),
\qquad
\alpha_{ij}=\operatorname{softmax}_{i\in\mathcal N(j)}(e_{ij}),
\\[4pt]
h'_j=\operatorname{Concat}_{k=1}^{H}\!\left(
\sum_{i\in\mathcal N(j)}\alpha_{ij}^{(k)}W^{(k)}h_i\right).
\end{gathered}
\]

Optional edge features add a learned edge term to \(e_{ij}\). Multi-head
outputs are concatenated.

Parameters
----------
in_dim : int | tuple[int, int]
    Input node feature dimension. Pass `(src_dim, dst_dim)` for bipartite
    source and destination features with different widths.
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
use_bias : bool, default=True
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
w_sender : Param
    Shared projection of shape `(in_dim, out_dim)` for an integer `in_dim`, or
    the source projection of shape `(src_dim, out_dim)` for paired dimensions.
w_receiver : Param | None
    Destination projection of shape `(dst_dim, out_dim)` for paired dimensions;
    otherwise `None` and `w_sender` is shared.
att_sender, att_receiver : Param
    Per-head attention vectors of shape `(num_heads, out_dim // num_heads)`.
b_out : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `use_bias=False`.
w_edge, att_edge : Param | None
    Edge projection and attention parameters. `None` unless `edge_dim` is set.

Example
-------
```python
num_nodes, num_edges = 3, 3
in_dim, out_dim, edge_dim = 16, 32, 8
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

gat = gnn.GATConv(in_dim, out_dim, num_heads=4, key=key)
y = gat(x, senders, receivers)  # (3, 16) -> (3, 32)

x_edge = jnp.ones((num_edges, edge_dim))
gat_edges = gnn.GATConv(in_dim, out_dim, num_heads=4, edge_dim=edge_dim, key=key)
y = gat_edges(x, senders, receivers, x_edge=x_edge)  # (3, 16) -> (3, 32)
```
