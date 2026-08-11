Pack graphs into one disconnected graph for batched message passing.

Node features are concatenated, edge indices are offset, and `graph_ids` maps
each node to its source graph. Call this outside `jax.jit` because its Python
sequences and output shapes vary with the batch.

Parameters
----------
xs : Sequence[jax.Array["n d", float]]
    Node feature matrix for each graph. Node counts may vary; feature
    dimensions must match.
senders : Sequence[jax.Array["e", int]]
    Sender indices for each graph. Edge counts may vary.
receivers : Sequence[jax.Array["e", int]]
    Receiver indices for each graph. Edge counts may vary.

Returns
-------
tuple[jax.Array["n d", float], jax.Array["e", int], jax.Array["e", int], jax.Array["n", int]]
    Concatenated node features, offset senders, offset receivers, and graph IDs.

Example
-------
```python
xs = [jnp.ones((3, 4)), jnp.ones((2, 4))]
senders = [jnp.array([0, 1]), jnp.array([0])]
receivers = [jnp.array([1, 2]), jnp.array([1])]

x, senders, receivers, graph_ids = gnn.batch_graphs(xs, senders, receivers)
# x: (5, 4)
# senders: [0, 1, 3], receivers: [1, 2, 4]
# graph_ids: [0, 0, 0, 1, 1]
```
