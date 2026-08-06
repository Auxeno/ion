Rebuild the graph with edges as nodes, joined where one edge ends and another begins.

Every edge of the input becomes a node of the line graph, and two of them are
joined when they form a two-hop path. Edge `a` connects to edge `b` when
`receivers[a] == senders[b]`, so together they run `i -> v -> k` through the
shared node `v`.

Because line-graph nodes are the rows of the input edge arrays, edge features
carry over as node features with no rearranging: pass `x_edge` straight into a
convolution alongside the returned arrays and it updates edge representations
using layers that only ever knew how to update nodes. The pivot node returned
alongside them is the natural place to attach line-graph edge features, either
`x[shared]` or a geometric quantity such as the angle between the two edges.

One node per row means an undirected graph stored with both directions gets a
separate node for each, so `(i, j)` and `(j, i)` hold independent state. This
is what lets a bond carry a different representation in each direction, as
DimeNet and ALIGNN rely on. Set `non_backtracking` to `False` to keep the pairs
that walk straight back down the reverse edge, which is usually wanted only
when the input is genuinely directed.

Self-loops make the join degenerate, since an edge `i -> i` chains with itself
and is its own reverse. Remove them with `remove_self_loops` first, and call
`to_undirected` beforehand if the input is undirected but not already stored
symmetrically. Parallel edges are not supported: with two identical rows there
is no unique reverse, so `non_backtracking` cannot be defined. Coalesce first
if the input may contain them.

The size of the result is `sum(indeg(v) * outdeg(v))` over nodes, which grows
with the square of the degree. This is affordable on the low-degree graphs
common in chemistry and can be very large on graphs with high-degree hubs. The
shape depends on the data either way, so this function cannot be called inside
`jax.jit`. Use it when preparing edge arrays, not inside a training step.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.
num_nodes : int
    Number of nodes in the graph. Every index must be less than this value.
non_backtracking : bool, default: True
    Drop pairs that return along the reverse of the incoming edge, so an
    edge never sends a message straight back where it came from.

Returns
-------
tuple[jax.Array["l", int], jax.Array["l", int], jax.Array["l", int]]
    Sender and receiver arrays over edge indices, followed by the node each
    connected pair of edges passes through.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 2, 3])
line_senders, line_receivers, shared = gnn.line_graph(senders, receivers, num_nodes=4)
# line_senders: [0, 0]
# line_receivers: [1, 2]
# shared: [1, 1]
```

Edge `0` runs `0 -> 1`, and both other edges leave node `1`, so it connects to
each of them through node `1`. Edges `1` and `2` end at leaves and connect to
nothing.

Running a convolution on the line graph updates the edge features:

```python
senders, receivers = gnn.remove_self_loops(senders, receivers)
senders, receivers, kept = gnn.to_undirected(senders, receivers, num_nodes)
x_edge = jnp.concatenate([x_edge, x_edge])[kept]

line_senders, line_receivers, shared = gnn.line_graph(senders, receivers, num_nodes)
x_edge = conv(x_edge, line_senders, line_receivers)
```

Batching commutes with this construction, because a disjoint union shares no
endpoints between graphs. Calling `batch_graphs` before or after gives the same
result.
