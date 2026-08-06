Sort edges by `(sender, receiver)` and drop duplicate rows.

An edge list is a sparse adjacency matrix in COO layout, and coalescing is the
sparse-matrix operation that puts one into canonical form: entries sorted, no
repeats. A duplicate row is a parallel edge, so the result is a simple graph
with edges in a deterministic order.

Sorting matters as much as the deduplication. It makes results independent of
how the data was loaded, groups each node's outgoing edges contiguously, and
lets a reverse-edge lookup use a binary search.

The third return value holds the index of the row kept for each surviving
edge, so edge features and masks can be filtered to match. Where duplicates
existed, the first occurrence in the input is the one kept. Reduce parallel
edges some other way by indexing with `kept` yourself.

The number of duplicates depends on the data, so the output shape is not known
ahead of time and this function cannot be called inside `jax.jit`. Use it when
preparing edge arrays, not inside a training step.

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
senders, receivers, kept = gnn.coalesce(senders, receivers)
# senders: [0, 1, 2]
# receivers: [1, 2, 0]
# kept: [1, 3, 0]
```

Edge features follow the same indices:

```python
senders, receivers, kept = gnn.coalesce(senders, receivers)
x_edge = x_edge[kept]
```

Self-loops are ordinary edges here and are preserved. Remove them first with
`remove_self_loops` if a simple loop-free graph is wanted.
