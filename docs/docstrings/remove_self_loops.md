Drop every edge whose sender and receiver are the same node.

All self-loops are removed, including any that were already present in the
input. The remaining edges keep their original order. The number of edges
removed depends on the data, so the output shape is not known ahead of time
and this function cannot be called inside `jax.jit`. Use it when preparing
edge arrays, not inside a training step.

Edge features and masks are not filtered. Apply the same `senders != receivers`
mask to them to keep the arrays aligned.

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
senders, receivers = gnn.remove_self_loops(senders, receivers)
# senders: [0, 1]
# receivers: [1, 2]
```

Filtering edge features alongside the edges:

```python
keep = senders != receivers
senders, receivers = gnn.remove_self_loops(senders, receivers)
x_edge = x_edge[keep]
```
