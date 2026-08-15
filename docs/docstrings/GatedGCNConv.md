Gated graph convolution ([Bresson and Laurent, 2017](https://arxiv.org/abs/1711.07553)).

Jointly updates nodes and edges. Each updated edge becomes a feature-wise gate
on its sender's message, normalized over the receiver's incoming edges:

\[
\tilde e_{ij} = W_e e_{ij} + W_s x_i + W_r x_j,
\qquad
x'_j = W_0 x_j +
\frac{\sum_{i \in \mathcal{N}(j)} \sigma(\tilde e_{ij}) \odot W_n x_i}
     {\sum_{i \in \mathcal{N}(j)} \sigma(\tilde e_{ij}) + \epsilon}.
\]

The layer returns \(\tilde e_{ij}\) as the updated edge representation. It does
not apply activation, normalization, or residual connections.

Parameters
----------
in_dim : int | tuple[int, int]
    Input node feature dimension. Pass `(src_dim, dst_dim)` for bipartite
    source and destination features with different widths.
out_dim : int
    Output dimension shared by node and edge features.
edge_dim : int
    Input edge feature dimension.
eps : float, default=1e-6
    Small value added to the normalized gate denominator.
use_bias : bool, default=True
    Whether to include separate node and edge output biases.
w_init : Initializer
    Weight initializer for all five transforms. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_root : Param
    Destination-root transform of shape `(dst_dim, out_dim)`.
w_neigh : Param
    Sender-message transform of shape `(src_dim, out_dim)`.
w_edge : Param
    Edge transform of shape `(edge_dim, out_dim)`.
w_sender : Param
    Sender contribution to the edge update, shape `(src_dim, out_dim)`.
w_receiver : Param
    Receiver contribution to the edge update, shape `(dst_dim, out_dim)`.
b_node : Param | None
    Node output bias of shape `(out_dim,)`. `None` when `use_bias=False`.
b_edge : Param | None
    Edge output bias of shape `(out_dim,)`. `None` when `use_bias=False`.
eps : float
    Value added to the gate denominator.

Example
-------
```python
num_nodes, num_edges, node_dim, edge_dim, out_dim = 3, 3, 4, 2, 8
x = jnp.ones((num_nodes, node_dim))
x_edge = jnp.ones((num_edges, edge_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

conv = gnn.GatedGCNConv(node_dim, out_dim, edge_dim=edge_dim, key=key)
x, x_edge = conv(x, senders, receivers, x_edge=x_edge)
x, x_edge = jax.nn.relu(x), jax.nn.relu(x_edge)
```
