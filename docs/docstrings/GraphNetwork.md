Graph Network block without global state ([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)).

Updates every edge from its incident nodes and optional current features, then
aggregates the updated edges and updates each destination node:

\[
e'_{ij} = \phi_e([x_i, x_j, e_{ij}]),
\qquad
x'_j = \phi_v([x_j, \rho(\{e'_{ij}\}_{i \in \mathcal N(j)})]).
\]

The updated edge representations are returned and, unless excluded from node
aggregation by an optional `edge_mask`, used as messages to the node update.
Omit `x_edge` to construct them from incident nodes alone.

Each node aggregates only its incoming edges, matching the paper's
\(E'_i = \{e'_k : r_k = i\}\). Some implementations also aggregate the edges a
node sends and pass both to the node update, which is a generalization of the
published block rather than the block itself. The two stages are exactly
[`EdgeUpdate`](#ion.gnn.EdgeUpdate) followed by
[`NodeUpdate`](#ion.gnn.NodeUpdate).

Parameters
----------
edge_model : Callable[[jax.Array], jax.Array]
    Update applied independently to each concatenated edge input.
node_model : Callable[[jax.Array], jax.Array]
    Update applied independently to each destination node concatenated with its
    aggregated incoming edges.
aggregate : Callable, default=segment_sum
    Edge-to-node reduction with the signature
    `aggregate(data, segment_ids, num_segments)`. Custom reductions must ignore
    segment IDs outside `[0, num_segments)`, as Ion's segment reductions do.

Attributes
----------
edge_model : Callable
    Edge update passed at construction.
node_model : Callable
    Node update passed at construction.
aggregate : Callable
    Edge-to-node reduction passed at construction.

Example
-------
```python
node_dim, edge_dim, message_dim, out_dim = 4, 2, 8, 6
edge_model = nn.MLP([2 * node_dim + edge_dim, 16, message_dim], key=key_edge)
node_model = nn.MLP([node_dim + message_dim, 16, out_dim], key=key_node)

network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
x, x_edge = network(x, senders, receivers, x_edge=x_edge)
```

For bipartite edges, pass `(x_src, x_dst)`. Sender indices select `x_src` rows,
receiver indices select `x_dst` rows, and only the destination nodes are
updated.
