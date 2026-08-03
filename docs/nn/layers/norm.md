# Normalization

`BatchNorm`, `LayerNorm`, and `RMSNorm` normalize feature values. `GroupNorm`
normalizes over channel groups and a configurable number of trailing spatial
dimensions. `SpectralNorm` normalizes a module parameter instead.

`BatchNorm` and `SpectralNorm` store their non-trainable state in an explicit
[`Buffers`](../../core/buffers.md) collection. Initialize it from the complete
model and keep the updated collection returned by training calls.

::: ion.nn.BatchNorm

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm

::: ion.nn.SpectralNorm
