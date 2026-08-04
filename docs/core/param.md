# Param

`Param` marks a JAX array as a model parameter. Ion uses it for parameter inspection and to tell the [`Optimizer`](optimizer.md) what it may update.

::: ion.nn.Param
    options:
      members: false

---

## Trainability

| Value | Meaning |
|---|---|
| `Param(array)` | Trainable parameter. |
| `Param(array, trainable=False)` | Frozen parameter. |
| Bare JAX array | Ordinary, non-parameter model data. |
| [`Buffer(array)`](buffers.md) | Non-trainable state mutated during a forward pass. |

Frozen params apply `jax.lax.stop_gradient`, receive no optimizer state, and remain identifiable as parameters. Freezing or unfreezing changes the pytree structure, so create a new optimizer afterwards.

## Array behaviour

`Param` implements JAX's array protocol, so it works in array expressions and proxies attributes such as `.shape`, `.dtype`, `.T`, and `.reshape(...)`. Arithmetic returns ordinary JAX arrays because intermediate results are not parameters.

```python
y = x @ w
w.shape
w.value
```

Use `param.value` when a function requires a plain array. It respects `stop_gradient` for frozen params; the private `param._value` does not.
