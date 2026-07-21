# Attention

Multi-head attention. `SelfAttention` attends within one sequence (optionally causal); `CrossAttention` attends from a query sequence into a separate context. Both accept an optional boolean `mask`; see [Conventions](../conventions.md#attention-masking) for mask shapes and patterns.

::: ion.nn.SelfAttention

::: ion.nn.CrossAttention
