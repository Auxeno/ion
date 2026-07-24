Append one self-loop edge for every node.

Existing edges are preserved, including existing self-loops. The returned
arrays therefore contain `num_nodes` additional entries.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
num_nodes : int
    Number of nodes in the graph.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int]]
    Sender and receiver arrays with self-loops appended.

Example
-------
```python
senders = jnp.array([0, 1])
receivers = jnp.array([1, 2])
senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes=3)
# senders: [0, 1, 0, 1, 2]
# receivers: [1, 2, 0, 1, 2]
```
