Wraps an optax `GradientTransformation` with Param-aware, pytree-native updates.

Parameters
----------
tx : optax.GradientTransformation or dict
    A single optax transform, or a dict mapping top-level model field names
    (or tuples of them) to per-field transforms.
model : PyTree
    The model to optimize. Frozen `Param`s and bare arrays are auto-partitioned
    out, so no optimizer state is allocated for them.

Attributes
----------
step : Array
    `int32` step counter, incremented once per `update` call.
state : PyTree
    Internal optax optimizer state.

Notes
-----
`update` returns a new model and a new optimizer; nothing is mutated in place. The optimizer snapshots the model's structure and trainability at construction, so changing either afterwards (via `freeze`/`unfreeze` or `at`) requires building a new optimizer.

Examples
--------
>>> optimizer = ion.Optimizer(optax.adam(3e-4), model)
>>> model, optimizer = optimizer.update(model, grads)

Per-field transforms (e.g. separate learning rates for a GAN):

>>> optimizer = ion.Optimizer(
...     {"generator": optax.adam(1e-4), "discriminator": optax.adam(4e-4)},
...     model,
... )
