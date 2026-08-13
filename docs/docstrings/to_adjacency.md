Scatter edges into a dense adjacency matrix.

Entry `(i, j)` is `1.0` when the edge `i -> j` is present and `0.0` otherwise,
so duplicate edges collapse. The result costs `num_nodes` squared memory; keep
it for small graphs and spectral preprocessing rather than message passing.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
num_nodes : int
    Number of nodes in the graph. Sets the size of the result.

Returns
-------
jax.Array["n n", float]
    Dense adjacency matrix.

Example
-------
```python
senders = jnp.array([0, 1, 2])
receivers = jnp.array([1, 2, 0])
adjacency = gnn.to_adjacency(senders, receivers, num_nodes=3)
# [[0., 1., 0.],
#  [0., 0., 1.],
#  [1., 0., 0.]]
```
