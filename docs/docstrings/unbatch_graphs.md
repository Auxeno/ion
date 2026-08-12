Split a batched graph back into its component graphs.

The inverse of `batch_graphs`. Node features are partitioned by `graph_ids`,
edges are assigned to the graph holding their sender, and edge indices are
shifted back to be local to each graph. Call this outside `jax.jit` because
its output shapes vary with the batch.

Parameters
----------
x : jax.Array["n d", float]
    Node features of the batched graph.
senders : jax.Array["e", int]
    Sender indices of the batched graph.
receivers : jax.Array["e", int]
    Receiver indices of the batched graph.
graph_ids : jax.Array["n", int]
    Source graph of each node. The number of graphs is `graph_ids.max() + 1`.

Returns
-------
tuple[list[jax.Array["n d", float]], list[jax.Array["e", int]], list[jax.Array["e", int]]]
    Per-graph node features, senders, and receivers.

Example
-------
```python
x = jnp.ones((5, 4))
senders = jnp.array([0, 1, 3])
receivers = jnp.array([1, 2, 4])
graph_ids = jnp.array([0, 0, 0, 1, 1])

xs, senders, receivers = gnn.unbatch_graphs(x, senders, receivers, graph_ids)
# xs: [(3, 4), (2, 4)]
# senders: [[0, 1], [0]], receivers: [[1, 2], [1]]
```
