# Normalization

Feature normalization layers. `LayerNorm` and `RMSNorm` normalize over the last dimension; `GroupNorm` normalizes over channel groups and a configurable number of trailing spatial dimensions. Ion ships no BatchNorm by design.

::: ion.nn.LayerNorm

::: ion.nn.RMSNorm

::: ion.nn.GroupNorm

---

## Why No BatchNorm?

BatchNorm carries running statistics that change during training. Ion modules
are immutable, so those statistics would need a separate update path whose
omission could silently leave evaluation using their initial values.
`LayerNorm` and `GroupNorm` do not require running statistics. Applications
that require BatchNorm can manage its statistics explicitly or use a library
with mutable model state.
