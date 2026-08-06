# Graph operations

Segment reductions, graph-level pooling, and graph-building helpers. These functions back the GNN layers and can also be composed into custom message-passing layers. The [GNN guide](guide.md) explains the array conventions and `graph_ids`.

Floating-point segment sum, mean, and softmax operations, including graph sum and mean pooling, compute in `float32` for numerical stability, then cast back to the input dtype.

`segment_max`, `segment_min`, and `segment_prod` are re-exported from `jax.ops`.

## Segment reductions

::: ion.gnn.segment_sum
    options:
      heading_level: 3

::: ion.gnn.segment_softmax
    options:
      heading_level: 3

::: ion.gnn.segment_mean
    options:
      heading_level: 3

## Graph pooling

::: ion.gnn.mean_pool
    options:
      heading_level: 3

::: ion.gnn.sum_pool
    options:
      heading_level: 3

::: ion.gnn.max_pool
    options:
      heading_level: 3

## Graph construction

::: ion.gnn.add_self_loops
    options:
      heading_level: 3

::: ion.gnn.remove_self_loops
    options:
      heading_level: 3

::: ion.gnn.degree
    options:
      heading_level: 3

::: ion.gnn.batch_graphs
    options:
      heading_level: 3
