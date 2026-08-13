Pooling by multi-head attention ([Lee et al., 2019](https://proceedings.mlr.press/v97/lee19d.html)).

Learns `num_seeds` query vectors that attend over the nodes of each graph. Nodes
supply keys and values, attention is normalized within a graph, and every seed
returns its own graph representation:

\[
r_{g,t} = W_o \sum_{i \in g}
\operatorname{softmax}_g\!\left(\frac{s_t^\top W_k x_i}{\sqrt{d_h}}\right) W_v x_i.
\]

The seeds are learned directly as queries, so the query projection of the paper's
attention block is absorbed into them. This is the bare pooling operation: the
residual connections, layer norms, and feedforwards of a full Set Transformer
block are not included, and stack around it with `nn.LayerNorm` and `nn.MLP`.

Parameters
----------
in_dim : int
    Input node feature dimension.
out_dim : int
    Output feature dimension per seed. Must be divisible by `num_heads`.
num_seeds : int, default=1
    Number of learned query vectors. Each returns one representation per graph.
num_heads : int, default=1
    Number of attention heads. Heads are concatenated into `out_dim`.
use_bias : bool, default=True
    Whether to include a learnable output bias.
w_init : Initializer
    Key, value, and output projection initializer. Glorot uniform by default.
seed_init : Initializer
    Seed query initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
seeds : Param
    Learned query vectors of shape `(num_seeds, out_dim)`.
w_k, w_v : Param
    Node key and value projections of shape `(in_dim, out_dim)`.
w_out : Param
    Output projection of shape `(out_dim, out_dim)`.
b_out : Param | None
    Output bias of shape `(out_dim,)`. `None` when `use_bias=False`.

Example
-------
```python
num_nodes, in_dim, out_dim = 5, 16, 32
x = jnp.ones((num_nodes, in_dim))
graph_ids = jnp.array([0, 0, 1, 1, 1])

pool = gnn.MultiHeadAttentionPool(in_dim, out_dim, num_seeds=4, num_heads=4, key=key)
graph_x = pool(x, graph_ids, num_graphs=2)  # (5, 16) -> (2, 4, 32)
graph_x = graph_x.reshape(2, -1)            # flatten the seeds into one vector
```

Note
----
Attention weights are normalized within each graph, so this readout is blind to
graph size in the way `mean_pool` is. Concatenate a `sum_pool` branch for tasks
that depend on counting nodes.
