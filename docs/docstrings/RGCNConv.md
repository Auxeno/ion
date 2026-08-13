Relational graph convolution layer ([Schlichtkrull et al., 2018](https://arxiv.org/abs/1703.06103)).

Gives each edge type its own neighbour transform, averaging within a relation
before summing across them:

\[
x'_i = W_s x_i + \sum_{r} \frac{1}{|\mathcal N_r(i)|}
\sum_{j \in \mathcal N_r(i)} W_r x_j.
\]

Relation types arrive as a per-edge index array rather than as separate graphs,
so one edge list carries every relation and each node is aggregated once.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output node feature dimension.
num_relations : int
    Number of edge types. Values in `edge_type` must be less than this.
num_bases : int | None, default=None
    If set, every relation transform becomes a learned mixture of `num_bases`
    shared matrices, holding the parameter count flat as relations grow.
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
w_neigh : Param
    Neighbour transforms of shape `(num_bases or num_relations, in_dim, out_dim)`.
w_coeff : Param | None
    Mixing coefficients of shape `(num_relations, num_bases)`. `None` when
    `num_bases` is not set and `w_neigh` holds one transform per relation.
w_self : Param
    Root transform of shape `(in_dim, out_dim)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `use_bias=False`.

Example
-------
```python
# Three-node cycle whose edges carry two relation types
num_nodes, in_dim, out_dim = 3, 16, 32
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
edge_type = jnp.array([0, 1, 0])

conv = gnn.RGCNConv(in_dim, out_dim, 2, key=key)
y = conv(x, senders, receivers, edge_type=edge_type)  # (3, 16) -> (3, 32)
```

```python
conv = gnn.RGCNConv(in_dim, out_dim, 12, num_bases=2, key=key)
```

Note
----
The forward pass projects every node under every relation before gathering the
one each edge needs, so it holds a `(num_nodes, num_relations, out_dim)`
intermediate. This suits a modest number of relations.
