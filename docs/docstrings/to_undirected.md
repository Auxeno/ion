Add the reverse of every edge so the graph is stored symmetrically.

The result is coalesced, so existing reverse edges and self-loops are not
duplicated. The returned indices address the original edges concatenated with
their reverses; use them to align copied or direction-dependent edge features.
Because the output size depends on the data, call this outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int], jax.Array["e2", int]]
    Symmetric sender and receiver arrays in coalesced order, followed by the
    index of the row kept for each edge, addressing the original edges
    concatenated with the reversed ones.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 0, 2])
senders, receivers, kept = gnn.to_undirected(senders, receivers)
# senders: [0, 1, 1, 2]
# receivers: [1, 0, 2, 1]
# kept: [0, 1, 2, 5]
```

Copy direction-independent features onto reverse edges:

```python
x_edge = jnp.concatenate([x_edge, x_edge])[kept]
```

Transform direction-dependent features as needed, for example by negating a
displacement:

```python
x_edge = jnp.concatenate([x_edge, -x_edge])[kept]
```
