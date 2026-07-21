# Optimizer

Wraps an optax `GradientTransformation` with Param-aware updates. It is a standalone pytree, so it works with `jax.jit` and `jax.lax.scan`.

::: ion.Optimizer
    options:
      members:
        - update

## How it works

**Auto-partitioning.** On construction, if the model contains any bare JAX arrays or frozen `Param` leaves, the optimizer wraps the transform with `optax.partition`, assigning the real optimizer to trainable params and `optax.set_to_zero()` to frozen params. No momentum or variance buffers are allocated for frozen weights, which matters for LoRA and other fine-tuning setups where most params are frozen.

**Update.** `update(model, grads)` calls the underlying optax transform, then applies the resulting deltas to the model's trainable `Param` leaves only. Non-`Param` arrays (like batch statistics) and frozen `Param` leaves pass through unchanged. The `Param` wrapper is preserved on updated values so trainability metadata survives the step. Both the new model and a new optimizer are returned; nothing is mutated in place.

**Step counter.** `step` is an `int32` array that increments on each `update` call, independent of any internal step tracking by optax transforms such as warmup schedules.

**Per-field transforms.** Pass a dict instead of a single transform to route different optimizers to different top-level model fields. Useful when components need independent learning rates or gradient clipping: GANs (generator vs discriminator), actor-critic RL (separate LR and grad-norm thresholds), or transfer learning (slow backbone, fast head).

```python
# gan.generator and gan.discriminator get separate optimizers
optimizer = ion.Optimizer(
    {"generator": optax.adam(1e-4), "discriminator": optax.adam(4e-4)},
    gan,
)

# Tuple keys group multiple fields under one transform
optimizer = ion.Optimizer(
    {
        ("actor", "std_raw"): optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4)),
        "critic": optax.chain(optax.clip_by_global_norm(1.5), optax.adam(1e-3)),
    },
    network,
)
```

Internally this uses `optax.partition` with labels derived from `jax.tree.map_with_path`. Every top-level field with trainable params must be covered by the dict; missing fields raise `ValueError` at construction. Frozen params within any group are still routed to `set_to_zero()`.

The optimizer snapshots the model's structure and trainability at construction, so changing either afterwards means building a new optimizer. See [Sharp Edges](../guides/sharp-edges.md).
