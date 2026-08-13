# Pooling

Learned attention-weighted pooling from node features to graph representations. `GlobalAttentionPool` scores each node with modules you supply: the `score` module decides how much each node contributes, the optional `value` module decides what features it contributes. `MultiHeadAttentionPool` instead learns its own query vectors, called seeds, and returns one representation per seed. For fixed mean, sum, and max readouts see [graph operations](../operations.md#pooling).

::: ion.gnn.GlobalAttentionPool

::: ion.gnn.MultiHeadAttentionPool
