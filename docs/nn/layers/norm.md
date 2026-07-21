# Normalization

Feature normalization layers. `LayerNorm` and `RMSNorm` normalize over the last dimension; `GroupNorm` normalizes over channel groups and a configurable number of trailing spatial dimensions (see [Reference](../reference.md#groupnorm-spatial-dimensions)). Ion ships no BatchNorm by design; the [Reference](../reference.md#why-no-batchnorm) page explains why.

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm
