Drop every edge whose sender and receiver are the same node.

The remaining edges keep their original order. The returned indices select the
same rows from edge features. Because the output size depends on the data, call
this outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int], jax.Array["e2", int]]
    Sender and receiver arrays with self-loops dropped, plus the retained
    indices into the original edge arrays.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 1, 2])
x_edge = jnp.array([10, 11, 12])
senders, receivers, kept = gnn.remove_self_loops(senders, receivers)
x_edge = x_edge[kept]
# senders: [0, 1]
# receivers: [1, 2]
# x_edge: [10, 12]
```
