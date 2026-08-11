Sort edges by `(sender, receiver)` and drop duplicate rows.

The first occurrence of each edge is kept. Use the returned indices to align
edge features or masks. Self-loops are preserved. Because the output size
depends on the data, call this outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int], jax.Array["e2", int]]
    Sorted sender and receiver arrays with duplicates dropped, followed by
    the index of the input row kept for each surviving edge.

Example
-------
```python
senders = jnp.array([2, 0, 2, 1])
receivers = jnp.array([0, 1, 0, 2])
x_edge = jnp.array([20, 10, 21, 12])
senders, receivers, kept = gnn.coalesce(senders, receivers)
x_edge = x_edge[kept]
# senders: [0, 1, 2]
# receivers: [1, 2, 0]
# kept: [1, 3, 0]
# x_edge: [10, 12, 20]
```
