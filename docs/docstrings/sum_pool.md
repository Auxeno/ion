Sum node features within each graph.

Sum pooling preserves node counts and is the graph-level readout used in the
GIN paper.

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
    Summed feature vector for each graph.

Example
-------
```python
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
graph_ids = jnp.array([0, 0, 1])
y = gnn.sum_pool(x, graph_ids, num_graphs=2)  # [[4., 6.], [10., 20.]]
```
