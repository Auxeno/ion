Build a graph whose nodes are the input edges.

Input edges `a` and `b` are joined when `receivers[a] == senders[b]`; the third
return value identifies that shared node. With `non_backtracking=True`, pairs
that immediately traverse a reverse edge are omitted.

Remove self-loops first, and coalesce parallel edges. For undirected graphs,
also call `to_undirected` if both directions are not already present. The
data-dependent output size means this function must run outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
non_backtracking : bool, default: True
    Drop pairs that return along the reverse of the incoming edge, so an
    edge never sends a message straight back where it came from.

Returns
-------
tuple[jax.Array["l", int], jax.Array["l", int], jax.Array["l", int]]
    Sender and receiver arrays over edge indices, followed by the node each
    connected pair of edges passes through.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 2, 3])
line_senders, line_receivers, shared = gnn.line_graph(senders, receivers)
# line_senders: [0, 0]
# line_receivers: [1, 2]
# shared: [1, 1]
```
