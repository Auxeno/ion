# Param

Wraps an array to mark it as a trainable or frozen parameter. JAX pytrees see all arrays equally and have no built-in way to distinguish trainable weights from frozen weights from plain buffers. `Param` makes this explicit.

::: ion.nn.Param

## How it works

A model field can be one of three things, and `Param` distinguishes the first two:

- `Param(array)`: trainable, gradients flow normally and optimizers update it.
- `Param(array, trainable=False)`: frozen, `stop_gradient` is applied so it is invisible to autodiff.
- a bare array: a plain data buffer, never treated as a parameter.

`Param` is registered as a pytree via `register_dataclass` with `_value` as a dynamic child (traced and differentiated by JAX) and `trainable` as static metadata (baked into compiled programs as a cache key). Changing `trainable` triggers recompilation, but it is a one-time flag set at construction.

`__jax_array__` returns the raw array for trainable params and applies `jax.lax.stop_gradient` for frozen params, making the `trainable` flag physically real in JAX's autodiff. `__getattr__` routes attribute access (`.shape`, `.dtype`, `.T`, `.reshape(...)`) through `jnp.asarray(self)`, so frozen params stay invisible to autodiff even through method calls. Arithmetic and comparisons return raw arrays, not `Param`, because intermediate results are not parameters.
