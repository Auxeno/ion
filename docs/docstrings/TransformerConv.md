Graph Transformer layer ([Shi et al., 2020](https://arxiv.org/abs/2009.03509)).

Applies scaled dot-product multi-head attention over incoming graph edges.
Receivers produce queries; senders produce keys and values. Optional edge
features are added to both keys and values. A learned projection of each
receiving node's features is added to the aggregated messages by default.

Parameters
----------
in_dim : int | tuple[int, int]
    Input node feature dimension. Pass `(src_dim, dst_dim)` for bipartite
    source and destination features with different widths.
out_dim : int
    Output node feature dimension. Must be divisible by `num_heads`.
num_heads : int, default=1
    Number of attention heads. Heads are concatenated into `out_dim`.
edge_dim : int | None, default=None
    Per-edge feature dimension. When set, one edge feature row is required for
    every directed edge.
use_root_weight : bool, default=True
    Add a learned linear projection of each receiving node's features to its
    aggregated messages.
use_beta : bool, default=False
    Use a learned sigmoid gate between the root projection and aggregated
    messages. Requires `use_root_weight=True`.
use_bias : bool, default=True
    Whether to include a learnable output bias.
w_init : Initializer
    Projection and gate weight initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_q, w_k, w_v : Param
    Destination-query, source-key, and source-value projections. Their first
    dimensions are `dst_dim`, `src_dim`, and `src_dim` respectively.
w_root : Param | None
    Projection for each destination node's own features, with shape
    `(dst_dim, out_dim)`. `None` when `use_root_weight=False`.
w_edge : Param | None
    Edge projection of shape `(edge_dim, out_dim)`. `None` unless `edge_dim`
    is set.
w_beta : Param | None
    Gate projection of shape `(3 * out_dim, 1)`. `None` unless `use_beta=True`.
b_out : Param | None
    Output bias of shape `(out_dim,)`. `None` when `use_bias=False`.

Example
-------
```python
num_nodes, num_edges = 3, 3
in_dim, out_dim, edge_dim = 16, 32, 8
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
x_edge = jnp.ones((num_edges, edge_dim))

conv = gnn.TransformerConv(
    in_dim,
    out_dim,
    num_heads=4,
    edge_dim=edge_dim,
    use_beta=True,
    key=key,
)
y = conv(x, senders, receivers, x_edge=x_edge)
```

```python
conv = gnn.TransformerConv((src_dim, dst_dim), out_dim, num_heads=4, key=key)
y_dst = conv((x_src, x_dst), senders, receivers)
```
