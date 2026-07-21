# Normalization

Feature normalization layers. `LayerNorm` and `RMSNorm` normalize over the last dimension; `GroupNorm` normalizes over channel groups and a configurable number of trailing spatial dimensions (see [Conventions](../conventions.md#groupnorm-spatial-dimensions)). Ion ships no BatchNorm by design; the [Conventions](../conventions.md#why-no-batchnorm) page explains why.

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm
