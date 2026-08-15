Build the node-induced subgraph within a number of hops of selected nodes.

The selected nodes appear first in their supplied order. Newly discovered nodes
follow, grouped by hop and ordered by their original node index. The returned
graph contains every edge between the discovered nodes. The data-dependent
output size means this function must run outside `jax.jit`.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
node_ids : jax.Array["s", int]
    Unique nodes from which to begin traversal.
num_hops : int
    Number of graph hops to traverse.
num_nodes : int
    Number of nodes in the original graph.
direction : str, default='in'
    Traverse incoming edges toward their senders, outgoing edges toward their
    receivers, or both. `"in"` follows the dependencies of Ion message passing.

Returns
-------
tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    Relabelled sender and receiver arrays, followed by the original node and
    edge indices. The first `node_ids.shape[0]` node rows correspond to the
    starting nodes.

Example
-------
```python
selected = jnp.array([4])
senders, receivers, node_ids, edge_ids = gnn.k_hop_subgraph(
    senders, receivers, selected, 2, num_nodes
)

x_sub = x[node_ids]
predictions = model(x_sub, senders, receivers)
node_predictions = predictions[:selected.shape[0]]
```
