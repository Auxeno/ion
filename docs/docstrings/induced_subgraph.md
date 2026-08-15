Build the node-induced subgraph over selected nodes.

Every edge whose sender and receiver are both selected is retained. Nodes are
relabelled in the order supplied, while retained edges preserve their original
order. The data-dependent output size means this function must run outside
`jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
node_ids : jax.Array["k", int]
    Unique node indices to retain, in the desired output order.
num_nodes : int
    Number of nodes in the original graph.

Returns
-------
tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    Relabelled sender and receiver arrays, followed by the original node and
    edge indices. Slice features with `x[node_ids]` and `x_edge[edge_ids]`.

Example
-------
```python
senders = jnp.array([0, 1, 2, 2, 3])
receivers = jnp.array([1, 2, 0, 3, 1])
selected = jnp.array([2, 0, 3])

senders, receivers, node_ids, edge_ids = gnn.induced_subgraph(
    senders, receivers, selected, num_nodes=4
)
x = x[node_ids]
x_edge = x_edge[edge_ids]
# senders: [0, 0]
# receivers: [1, 2]
# node_ids: [2, 0, 3]
# edge_ids: [2, 3]
```
