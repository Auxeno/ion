Drop every edge whose sender and receiver are the same node.

The remaining edges keep their original order. Edge features are not filtered;
apply the same `senders != receivers` mask to keep them aligned. Because the
output size depends on the data, call this outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int]]
    Sender and receiver arrays with self-loops dropped.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 1, 2])
x_edge = jnp.array([10, 11, 12])
keep = senders != receivers
senders, receivers = gnn.remove_self_loops(senders, receivers)
x_edge = x_edge[keep]
# senders: [0, 1]
# receivers: [1, 2]
# x_edge: [10, 12]
```
