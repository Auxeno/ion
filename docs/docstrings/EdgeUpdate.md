Edge update from the Graph Network framework ([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)).

Concatenates each edge's sender features, receiver features, and current edge
features, then applies a caller-supplied model:

\[
e'_{ij} = \phi_e([x_i, x_j, e_{ij}]).
\]

Parameters
----------
edge_model : Module
    Update network applied independently to every concatenated edge input.
    Supplies all of the layer's weights, so `EdgeUpdate` takes no `key` and
    creates none of its own.

Attributes
----------
edge_model : Module
    The update network passed at construction.

Example
-------
```python
num_nodes, num_edges, node_dim, edge_dim, out_dim = 3, 3, 4, 2, 8
x = jnp.ones((num_nodes, node_dim))
x_edge = jnp.ones((num_edges, edge_dim))
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])

edge_model = nn.MLP([2 * node_dim + edge_dim, 16, out_dim], key=key)
update = gnn.EdgeUpdate(edge_model)
x_edge = update(x, senders, receivers, x_edge=x_edge)  # (3, 2) -> (3, 8)
```
