Node update from the Graph Network framework ([Battaglia et al., 2018](https://arxiv.org/abs/1806.01261)).

Aggregates edge features selected by an optional `edge_mask` at their receivers,
concatenates the result with each current destination node, and applies a
caller-supplied model:

\[
x'_j = \phi_v([x_j, \rho(\{m_{ij}\}_{i \in \mathcal N(j)})]).
\]

Parameters
----------
node_model : Callable[[jax.Array], jax.Array]
    Update applied independently to each destination node concatenated with its
    aggregated incoming edge features.
aggregate : Callable, default=segment_sum
    Edge-to-node reduction with the signature
    `aggregate(data, segment_ids, num_segments)`. Custom reductions must ignore
    segment IDs outside `[0, num_segments)`, as Ion's segment reductions do.

Attributes
----------
node_model : Callable
    Node update passed at construction.
aggregate : Callable
    Message-to-node reduction passed at construction.

Example
-------
```python
node_model = nn.MLP([node_dim + edge_dim, hidden_dim, out_dim], key=key)
update = gnn.NodeUpdate(node_model, aggregate=gnn.segment_mean)
x = update(x, senders, receivers, x_edge=x_edge)
```
