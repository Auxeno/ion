Pack graphs into one disconnected graph for batched message passing.

Node features are concatenated and each graph's edge indices are offset by the
number of preceding nodes. The returned `graph_ids` maps every node back to its
source graph. Call this function outside `jax.jit` because its Python
sequences and output shapes vary with the batch.

Parameters
----------
xs : Sequence[jax.Array["_ d", float]]
    Node feature matrix for each graph. Feature dimensions must match.
senders : Sequence[jax.Array["_", int]]
    Sender indices for each graph.
receivers : Sequence[jax.Array["_", int]]
    Receiver indices for each graph.

Returns
-------
tuple[jax.Array, jax.Array, jax.Array, jax.Array]
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
