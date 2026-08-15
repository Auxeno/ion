Graph Isomorphism Network layer with edge features ([Hu et al., 2020](https://arxiv.org/abs/1905.12265)).

Adds edge features to each message before a ReLU, then sum-aggregates and applies
a caller-supplied MLP:

\[
h_i' = \operatorname{MLP}\!\left((1 + \epsilon)h_i +
\sum_{j \in \mathcal{N}(i)} \operatorname{ReLU}(h_j + e_{ji})\right).
\]

Edge features share the node feature dimension, so embed them to that width
first. Bipartite source, destination, and edge features must all have the same
width because they are added before the MLP.

Parameters
----------
mlp : Callable[[jax.Array], jax.Array]
    Update network applied after aggregation. Supplies all of the layer's
    weights, so `GINEConv` takes no `key` and creates none of its own.
eps : float, default=0.0
    Weights a node's own features against its aggregated neighbours. Fixed unless
    `train_eps=True`.
train_eps : bool, default=False
    If `True`, `eps` becomes a learnable scalar `Param`.

Attributes
----------
mlp : Callable[[jax.Array], jax.Array]
    The update network passed at construction.
eps : Param | float
    Learnable scalar when `train_eps=True`, otherwise the fixed float.

Example
-------
```python
num_nodes, num_edges, in_dim, hidden_dim, out_dim = 3, 3, 16, 32, 32
x = jnp.ones((num_nodes, in_dim))
x_edge = jnp.ones((num_edges, in_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

gine = gnn.GINEConv(nn.MLP([in_dim, hidden_dim, out_dim], key=key))
y = gine(x, senders, receivers, x_edge=x_edge)  # (3, 16) -> (3, 32)
```
