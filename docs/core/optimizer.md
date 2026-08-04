# Optimizer

`Optimizer` wraps an Optax transform and applies it to trainable [`Param`](param.md) leaves. Frozen params, bare arrays, and [`Buffer`](buffers.md) state are left unchanged.

::: ion.Optimizer
    options:
      members:
        - update

---

## Updating a model

Construct an optimizer from an Optax transform and the model it will update:

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)

grads = jax.grad(loss_fn)(model, x, y)
model, optimizer = optimizer.update(model, grads)
```

The optimizer is a JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html), so the whole step works with `jax.jit` and `jax.lax.scan`.

## Per-field transforms

Pass a dictionary to assign different transforms to top-level model fields:

```python
optimizer = ion.Optimizer(
    {
        ("actor", "std_raw"): optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)),
        "critic": optax.chain(optax.clip_by_global_norm(1.5), optax.adam(1e-3)),
    },
    network,
)
```

Tuple keys group multiple fields under one transform. Every top-level field containing trainable parameters must be covered; missing fields raise `ValueError`.

## Structural changes

Create a new optimizer after changing the model's structure or param trainability:

```python
model = model.freeze()
model = model.at.head.set(model.head.unfreeze())
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

The optimizer checks this on every update and raises `ValueError` if the structure no longer matches. Frozen params and bare arrays receive no optimizer state.
