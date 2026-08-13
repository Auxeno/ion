Gather the nonzero entries of a dense adjacency matrix into edges.

Edges come out sorted by `(sender, receiver)`. Without `num_edges` the output
size depends on the data, so call it outside `jax.jit`. Passing `num_edges`
fixes the size: spare slots hold the out-of-range index `num_nodes`, which
segment reductions drop but a plain gather clamps to the last node.

Parameters
----------
adjacency : jax.Array["n n", float]
    Dense adjacency matrix. Any nonzero entry counts as an edge.
num_edges : int | None, default: None
    Number of edges to return, required under `jax.jit`. Extra edges beyond
    this count are dropped.

Returns
-------
tuple[jax.Array["e", int], jax.Array["e", int]]
    Sender and receiver arrays.

Example
-------
```python
adjacency = jnp.array([[0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.]])

senders, receivers = gnn.from_adjacency(adjacency)
# senders: [0, 1, 2]
# receivers: [1, 2, 0]

# Room for four edges, so the spare slot is padded with the index 3
senders, receivers = gnn.from_adjacency(adjacency, num_edges=4)
# senders: [0, 1, 2, 3]
# receivers: [1, 2, 0, 3]
```
