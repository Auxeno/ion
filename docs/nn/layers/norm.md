# Normalization

`BatchNorm`, `LayerNorm`, and `RMSNorm` normalize feature values. `GroupNorm` normalizes over channel groups and a configurable number of trailing spatial dimensions. `SpectralNorm` normalizes a module parameter instead.

`BatchNorm` and `SpectralNorm` hold their non-trainable state in [`Buffer`](../../core/buffers.md) fields, updated in place during training calls.

`BatchNorm`, `LayerNorm`, `GroupNorm`, and `RMSNorm` compute in `float32` for numerical stability, then cast back to the input dtype.

::: ion.nn.BatchNorm

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm

::: ion.nn.SpectralNorm
