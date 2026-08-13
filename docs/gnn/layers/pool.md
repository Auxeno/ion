# Pooling

Learned attention-weighted pooling from node features to graph representations. The `score` module decides how much each node contributes; the optional `value` module decides what features it contributes. For fixed mean, sum, and max readouts see [graph operations](../operations.md#pooling).

::: ion.gnn.GlobalAttentionPool
