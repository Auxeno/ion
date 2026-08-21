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

## Inspection

Printing an optimizer shows the transforms it was built from and the state they carry:

```python
optimizer = ion.Optimizer(
    optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=0.1)),
    model,
)
```

--8<-- "docs/assets/optimizer-repr.html"

A scheduled learning rate is reported at the step the optimizer has reached, rather than by the schedule's name:

```python
schedule = optax.warmup_cosine_decay_schedule(0.0, 3e-4, 100, 1000)
optimizer = ion.Optimizer(optax.adamw(schedule, weight_decay=0.1), model)
```

--8<-- "docs/assets/optimizer-schedule-repr.html"

Inside `jax.jit` the step is abstract, so no value is read from it and the rate is left out.

`step` counts calls to `update`, and is the only counter every optimizer has: plain SGD, RMSProp and AdaGrad keep no count of their own. Where a transform does carry one it is listed only when it differs from `step`, which happens under gradient accumulation, where Adam's `count` tracks applied updates rather than calls.

## Structural changes

Create a new optimizer after changing the model's structure or param trainability:

```python
model = model.freeze()
model = model.at.head.set(model.head.unfreeze())
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

The optimizer checks this on every update and raises `ValueError` if the structure no longer matches. Frozen params and bare arrays receive no optimizer state.
