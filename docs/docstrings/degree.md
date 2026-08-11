Count how many edges reference each node.

Pass `senders` for out-degree or `receivers` for in-degree. Isolated nodes
receive zero; the result is an integer array of fixed length `num_nodes` and
works inside `jax.jit`.

Parameters
----------
indices : jax.Array["e", int]
    Node index for each edge, either `senders` or `receivers`.
num_nodes : int
    Number of nodes in the graph. Sets the length of the result.

Returns
-------
jax.Array["n", int]
    Number of edges referencing each node.

Example
-------
```python
senders = jnp.array([0, 0, 1])
receivers = jnp.array([1, 2, 2])

out_degree = gnn.degree(senders, num_nodes=3)
# [2, 1, 0]

in_degree = gnn.degree(receivers, num_nodes=3)
# [0, 1, 2]
```
