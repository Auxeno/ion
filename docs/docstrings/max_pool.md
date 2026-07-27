Maximum node features within each graph.

Empty graphs return zeros rather than the `-inf` fill used by
`gnn.segment_max`.

Parameters
----------
x : jax.Array["n d", float]
    Node feature matrix.
graph_ids : jax.Array["n", int]
    Graph index for each node.
num_graphs : int
    Total number of graphs.

Returns
-------
jax.Array["g d", float]
    Elementwise maximum feature vector for each graph.

Example
-------
```python
x = jnp.array([[1.0, 4.0], [3.0, 2.0], [10.0, 20.0]])
graph_ids = jnp.array([0, 0, 1])
y = gnn.max_pool(x, graph_ids, num_graphs=2)  # [[3., 4.], [10., 20.]]
```
