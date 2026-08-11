Global attention pooling ([Li et al., 2019](https://proceedings.mlr.press/v97/li19d.html)).

Scores each node, normalizes the scores within its graph, and takes an
attention-weighted sum of node features:
\(r_g = \sum_{i \in g} \operatorname{softmax}_g(a_i) f(x_i)\).

Parameters
----------
score : Module
    Module deciding how much each node contributes. It maps every node feature
    row to one importance logit and must return shape `(num_nodes, 1)`.
value : Module | None, default=None
    Optional module deciding what each node contributes. It maps node features
    to the values used in the weighted sum. When `None`, the input features are
    pooled directly.

Attributes
----------
score : Module
    Node scoring module supplied at construction.
value : Module | None
    Node value module supplied at construction.

Example
-------
```python
num_nodes, in_dim, out_dim, num_graphs = 5, 16, 32, 2
x = jnp.ones((num_nodes, in_dim))
graph_ids = jnp.array([0, 0, 1, 1, 1])
key_score, key_value = jax.random.split(key)

pool = gnn.GlobalAttentionPool(
    score=nn.Linear(in_dim, 1, use_bias=False, key=key_score),
    value=nn.Linear(in_dim, out_dim, key=key_value),
)
graph_x = pool(x, graph_ids, num_graphs)  # (5, 16) -> (2, 32)
```

Note
----
`GlobalAttentionPool` creates no parameters; `score` and `value` own them. A
bias on a linear `score` has no effect because graph-wise softmax is invariant
to a constant shift.
