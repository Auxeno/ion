# Normalization

Feature normalization layers. `LayerNorm` and `RMSNorm` normalize over the last dimension; `GroupNorm` normalizes over channel groups and a configurable number of trailing spatial dimensions. Ion ships no BatchNorm by design.

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm

---

## Why No BatchNorm or SpectralNorm?

BatchNorm carries running statistics that change during training, and
SpectralNorm carries a power-iteration estimate updated on every forward pass.
Ion modules are immutable, so that state would need a separate update path whose
omission could silently leave evaluation using its initial values. `LayerNorm`
and `GroupNorm` do not require running state. Applications that need BatchNorm or
SpectralNorm can manage the state explicitly or use a library with mutable model
state.
