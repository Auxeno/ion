# GNN Ops

Segment reductions, graph-level pooling, and graph-building helpers. These
functions back the GNN layers and can also be composed into custom
message-passing layers. They use the array conventions and `graph_ids`
described in the [GNN reference](index.md).

The `jax.ops` functions `segment_sum`, `segment_max`, `segment_min`, and
`segment_prod` are also re-exported from `ion.gnn`, so all segment reductions
share the same namespace.

## Segment Reductions

::: ion.gnn.segment_softmax

::: ion.gnn.segment_mean

## Graph Pooling

::: ion.gnn.mean_pool

::: ion.gnn.sum_pool

::: ion.gnn.max_pool

## Graph Construction

::: ion.gnn.add_self_loops

::: ion.gnn.batch_graphs
