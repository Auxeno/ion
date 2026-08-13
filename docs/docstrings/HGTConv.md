Heterogeneous graph transformer layer ([Hu et al., 2020](https://arxiv.org/abs/2003.01332)).

Projects each node under its own type, then scores every edge with a matrix and a
learned prior belonging to its relation:

\[
\alpha_{s,e,t} = \operatorname*{softmax}_{s \in \mathcal N(t)}
\left(\frac{K(s)\,W^{A}_{\phi(e)}\,Q(t)^{\top}\,\mu_{\phi(e)}}{\sqrt{d}}\right).
\]

Messages carry a second per-relation matrix, and the aggregate is mixed with the
node's input through a learned per-type gate \(g_{\tau(t)}\):

\[
x'_t = g_{\tau(t)}\,W^{O}_{\tau(t)}\,\operatorname{gelu}\!\left(
\sum_{s \in \mathcal N(t)} \alpha_{s,e,t}\,V(s)\,W^{M}_{\phi(e)}\right)
+ \left(1 - g_{\tau(t)}\right) x_t.
\]

The softmax runs over all of a node's incoming edges at once, so relations compete
for attention rather than being normalized separately and combined afterwards.
Node types and relations are index arrays rather than separate graphs, so a single
node feature matrix and edge list describe the whole heterogeneous graph, and every
node type shares one input width.

Parameters
----------
in_dim : int
    Input node feature dimension, shared by every node type.
out_dim : int
    Output node feature dimension. Must be divisible by `num_heads`.
num_node_types : int
    Number of node types. Values in `node_type` must be less than this.
num_relations : int
    Number of edge types. Values in `edge_type` must be less than this.
num_heads : int, default=1
    Number of attention heads, each producing `out_dim // num_heads` features.
use_skip : bool, default=True
    Whether to mix the output with the input through a learned per-type gate.
    Requires `in_dim` to equal `out_dim`, since the two are combined directly.
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
w_q, w_k, w_v : Param
    Per-node-type query, key, and value projections of shape
    `(num_node_types, in_dim, out_dim)`.
w_att : Param
    Per-relation attention matrices of shape
    `(num_relations, num_heads, head_dim, head_dim)`.
w_msg : Param
    Per-relation message matrices, shaped like `w_att`.
w_out : Param
    Per-node-type output projections of shape `(num_node_types, out_dim, out_dim)`.
mu : Param
    Attention prior per relation and head, of shape `(num_relations, num_heads)`.
    Initialized to ones.
skip : Param | None
    Pre-sigmoid gate per node type, of shape `(num_node_types,)`. Initialized to
    ones. `None` when `use_skip=False`.
b_out : Param | None
    Per-node-type bias of shape `(num_node_types, out_dim)`. `None` when
    `use_bias=False`.

Example
-------
```python
# Four nodes over two types, joined by edges of two relations
num_nodes, num_node_types, num_relations, dim = 4, 2, 2, 32
x = jnp.ones((num_nodes, dim))
node_type = jnp.array([0, 1, 0, 1])
senders = jnp.array([0, 1, 2, 3])
receivers = jnp.array([1, 2, 3, 0])
edge_type = jnp.array([0, 1, 0, 1])

conv = gnn.HGTConv(dim, dim, num_node_types, num_relations, num_heads=4, key=key)
y = conv(x, senders, receivers, node_type=node_type, edge_type=edge_type)  # (4, 32)
```

Note
----
Node types usually differ in width, so encode each into the shared `in_dim` first,
for example with one `nn.Linear` per type. The forward pass projects every node
under every type and every relation before gathering, holding
`(num_nodes, num_node_types, out_dim)` and `(num_nodes, num_relations, out_dim)`
intermediates, which suits a modest number of types and relations.
