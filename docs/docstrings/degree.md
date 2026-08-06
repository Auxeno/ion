Count how many edges reference each node.

The function counts occurrences of each node index in a single array, so the
array you pass selects which degree you get. Pass `senders` for out-degree and
`receivers` for in-degree. The two agree when an undirected graph is stored
with both directions of every edge, and differ otherwise.

Nodes with no incident edges get a count of zero. The result is an integer
array; cast it before use in floating-point normalization.

Unlike `remove_self_loops`, the output shape is fixed by `num_nodes`, so this
function works inside `jax.jit`.

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

Degree-based normalization casts the counts to a floating-point dtype:

```python
in_degree = gnn.degree(receivers, num_nodes).astype(x.dtype)
norm = jnp.where(in_degree > 0, lax.rsqrt(in_degree), 0.0)
```
