Graph Isomorphism Network layer ([Xu et al., 2019](https://arxiv.org/abs/1810.00826)).

Sum-aggregates neighbor features and applies a caller-supplied MLP:

\[
h_i' = \operatorname{MLP}\!\left((1 + \epsilon)h_i +
\sum_{j \in \mathcal{N}(i)} h_j\right).
\]

Sum aggregation preserves neighbor multiplicity, making GIN as discriminative
as the Weisfeiler-Lehman graph isomorphism test.

Parameters
----------
mlp : Module
    Update network applied after aggregation. Supplies all of the layer's
    weights, so `GINConv` takes no `key` and creates none of its own.
eps : float, default=0.0
    Weights a node's own features against its aggregated neighbors. Fixed unless
    `train_eps=True`.
train_eps : bool, default=False
    If `True`, `eps` becomes a learnable scalar `Param`.

Attributes
----------
mlp : Module
    The update network passed at construction.
eps : Param | float
    Learnable scalar when `train_eps=True`, otherwise the fixed float.

Example
-------
```python
num_nodes, in_dim, hidden_dim, out_dim = 3, 16, 32, 32
x = jnp.ones((num_nodes, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

gin = gnn.GINConv(nn.MLP([in_dim, hidden_dim, out_dim], key=key))
y = gin(x, senders, receivers)  # (3, 16) -> (3, 32)

gin_trainable = gnn.GINConv(nn.MLP([in_dim, hidden_dim, out_dim], key=key), train_eps=True)
```

The layer also accepts `(x_src, x_dst)` for bipartite message passing and
returns one row per destination node. Source and destination feature widths
must match because their representations are added before the MLP.
