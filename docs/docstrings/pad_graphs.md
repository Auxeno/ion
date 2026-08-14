Pad a batched graph to fixed node, edge, and graph capacity.

Spare edges take the sender and receiver index `num_nodes`, and spare nodes take
the graph id `num_graphs`, one past the last real entry in each case. Segment
reductions drop out-of-range indices, so padding never reaches a real node and
pooling returns one row per real graph with nothing to mask or slice off.

Every batch padded to the same capacity has the same shapes, so a compiled step
is traced once and reused. Call this outside `jax.jit`, after `batch_graphs`:
padding happens on the host, keeping the varying widths away from XLA, which
would otherwise compile a separate kernel for each one.

Parameters
----------
x : jax.Array["n d", float]
    Batched node features.
senders : jax.Array["e", int]
    Batched sender indices.
receivers : jax.Array["e", int]
    Batched receiver indices.
graph_ids : jax.Array["n", int]
    Graph index of each node, as returned by `batch_graphs`.
num_nodes : int
    Node capacity. Fewer nodes than the batch contains raises.
num_edges : int
    Edge capacity. Fewer edges than the batch contains raises.
num_graphs : int
    Graph capacity, and the value to pass to the pooling call.

Returns
-------
tuple[jax.Array["n d", float], jax.Array["e", int], jax.Array["e", int], jax.Array["n", int]]
    Padded node features, senders, receivers, and graph IDs.

Example
-------
```python
x, senders, receivers, graph_ids = gnn.batch_graphs(xs, senders_list, receivers_list)
x, senders, receivers, graph_ids = gnn.pad_graphs(
    x, senders, receivers, graph_ids, num_nodes=512, num_edges=2048, num_graphs=32
)

h = conv(x, senders, receivers)
graph_h = gnn.mean_pool(h, graph_ids, num_graphs=32)
# graph_h: (32, d), one row per real graph
```
