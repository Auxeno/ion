# Param

Wraps an array to mark it as a trainable or frozen model parameter. Bare arrays
remain ordinary model data rather than parameters.

::: ion.nn.Param
    options:
      members: false

---

## Trainability

A model field can represent three different things:

- `Param(array)` is trainable: gradients flow and the optimizer updates it.
- `Param(array, trainable=False)` is frozen: `stop_gradient` produces zero
  gradients and no optimizer state is allocated.
- A bare array is ordinary model data: it participates in computation but is
  not treated as a parameter.

Values updated by stateful layers are [`Buffer`](buffers.md)s, which are
mutable and contribute no pytree leaves.

The `trainable` flag is static [pytree](https://docs.jax.dev/en/latest/pytrees.html) metadata. Freezing or unfreezing changes
the pytree structure and therefore requires a new `Optimizer`.

## Array Behaviour

`Param` implements JAX's array protocol, so it works in array expressions and
proxies attributes such as `.shape`, `.dtype`, `.T`, and `.reshape(...)`.
Arithmetic returns ordinary JAX arrays because intermediate results are not
parameters.

```python
y = x @ w
w.shape
jnp.asarray(w)
```

Use `jnp.asarray(param)` when the underlying array is needed. Accessing the
private `param._value` bypasses `stop_gradient` for frozen parameters.

## How does it work?

`Param` is a JAX pytree with its array as a dynamic child and `trainable` as
static metadata. Its array protocol returns the underlying array for trainable
parameters and applies `jax.lax.stop_gradient` for frozen parameters.

Arithmetic and attribute access pass through that protocol, which keeps frozen
parameters outside autodiff even when using operations such as `.reshape()` or
`.T`.
