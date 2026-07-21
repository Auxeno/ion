# GNN Ops

Segment reductions, graph-level pooling, and graph-building helpers. These back the GNN layers but are also the toolkit for writing custom message-passing layers. See [GNN](index.md) for the array conventions and `graph_ids`.

## segment_softmax

Softmax normalized within segments. Used internally by GATConv and GATv2Conv to normalize attention weights per receiver node, but useful for custom GNN layers too.

```python
from ion.gnn import segment_softmax

# Normalize scores so they sum to 1 per receiver node
weights = segment_softmax(scores, receivers, num_nodes)
```

## segment_mean

Mean of data within each segment. JAX ships `segment_sum`, `segment_max`, `segment_min` and `segment_prod` but no mean; this fills the gap. Empty segments give zeros, not NaN.

```python
from ion.gnn import segment_mean

means = segment_mean(messages, receivers, num_nodes)
```

The four `jax.ops` segment reductions are also re-exported from `ion.gnn`, so every segment reduction is reachable as `gnn.segment_*` alongside `segment_softmax` and `segment_mean`.

## Pooling: mean_pool, sum_pool, max_pool

Graph-level readout for graph classification. Each pools node features `(n, d)` into per-graph vectors `(g, d)` using a `graph_ids` array that maps each node to its graph (see [Batching Multiple Graphs](index.md#batching-multiple-graphs)).

```python
from ion.gnn import mean_pool

g = mean_pool(x, graph_ids, num_graphs)  # (n, d) -> (g, d)
```

`max_pool` returns zeros for empty graphs rather than `segment_max`'s `-inf` fill; `mean_pool` likewise guards empty graphs to zeros. Sum pooling is the readout used in the GIN paper, since it preserves node counts.

## add_self_loops

Appends self-loop edges (i -> i) for every node.

```python
from ion.gnn import add_self_loops

senders, receivers = add_self_loops(senders, receivers, num_nodes)
# senders and receivers now have num_nodes extra entries
```

## batch_graphs

Packs a list of graphs into a single disconnected graph for batched message passing, returning a `graph_ids` array for per-graph pooling. See [Batching Multiple Graphs](index.md#batching-multiple-graphs) for the full walkthrough.
