# Optimizer

Wraps an optax `GradientTransformation` with `Param`-aware, immutable updates.
The optimizer is a JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html) and can be threaded directly through `jax.jit`
and `jax.lax.scan`.

::: ion.Optimizer
    options:
      members:
        - update

---

## Per-field Transforms

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

Tuple keys group multiple fields under one transform. Every top-level field
containing trainable parameters must be covered; missing fields raise
`ValueError`.

## Structural Changes

The optimizer records the model's pytree structure and trainability when it is
constructed. Create a new optimizer after changing either:

```python
model = model.freeze()
model = model.at.head.set(model.head.unfreeze())
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

Reusing an optimizer with a structurally different model raises `ValueError`
before applying updates.

## How does it work?

The optimizer partitions the model when it is constructed. Trainable
parameters use the supplied optax transform, while frozen parameters and bare
arrays are routed to `optax.set_to_zero()`. This avoids allocating momentum or
variance buffers for values that cannot be updated. If every leaf is
trainable, no partition is added.

`update` asks optax for parameter deltas, applies them only to trainable
`Param` leaves, and returns a new model and optimizer. The `Param` wrappers and
their trainability metadata are preserved.
