# Internals

Everything behind the scenes in Ion. Three files and <1000 lines of code make up the whole engine. This document explains the design. Readers are encouraged to check out the source code:

- [`ion/nn/param.py`](../ion/nn/param.py): Param wrapper, trainable/frozen distinction
- [`ion/nn/module.py`](../ion/nn/module.py): Module base class, pytree registration
- [`ion/optimizer.py`](../ion/optimizer.py): Optimizer wrapper, auto-partitioning for frozen params

## Param (`ion/nn/param.py`)

JAX pytrees see all arrays equally and have no built-in way to distinguish trainable weights from frozen weights from plain buffers. `Param` makes this explicit:

- `Param(array)`: trainable (gradients flow normally, optimizers update)
- `Param(array, trainable=False)`: frozen (`stop_gradient` applied via `__jax_array__`, making this parameter invisible to autodiff)
- bare array: plain data buffer, never treated as a parameter

### Pytree registration

`Param` is registered via `register_dataclass` with `_value` as a dynamic child (traced/differentiated by JAX) and `trainable` as static metadata (baked into compiled programs as a cache key). Changing `trainable` triggers recompilation, but it's a one-time flag set at construction.

### Transparent array behavior

`__jax_array__` returns the raw array for trainable params and applies `jax.lax.stop_gradient` for frozen params, making the `trainable` flag physically real in JAX's autodiff. `__getattr__` routes attribute access (`.shape`, `.dtype`, `.T`, `.reshape(...)`) through `jnp.asarray(self)`, which calls `__jax_array__()`, so frozen params remain invisible to autodiff even through method calls. The `_value` field is private and should not be accessed directly in user code, as it bypasses `stop_gradient`. Arithmetic and comparisons return raw arrays, not `Param`, because intermediate results are not parameters.

## Module (`ion/nn/module.py`)

JAX requires two things from objects in `jit`/`grad`/`vmap`: pytree registration so JAX can traverse their structure, and immutability so tracing produces correct results. Plain Python classes satisfy neither.

Three things happen in `__init_subclass__` when a class inherits from `Module`:

1. **Dataclass conversion.** `@dataclasses.dataclass` is applied. If the
   subclass defines its own `__init__`, it is kept; otherwise one is generated
   from the annotations.

2. **Pytree registration.** The class is registered with `register_pytree_with_keys`. Each field is classified once at construction time (via `isinstance` checks) and the result is cached on the instance:

   - **Array-like** (`Param`, `Module`, `jax.Array`, `np.ndarray`) → dynamic child, passed to JAX as-is.
   - **Container with array-like content at any depth** (a tuple of `Module`s in `Sequential`, a `list[list[Module]]`) → dynamic child. Containers are inspected recursively via `jax.tree.leaves`. Pure containers (only modules and arrays) traverse natively; mixed containers have their non-array elements wrapped in `_Static` at any nesting depth so JAX treats them as compile-time constants.
   - **Everything else** (int, float, str, callable, None, containers with no arrays anywhere, ...) → static auxiliary data, stored in the treedef directly. No wrapping needed.

   Since modules are frozen after `__init__`, the classification never changes and subsequent flatten calls skip the `isinstance` checks entirely. Unflatten restores the cached classification from the treedef's auxiliary data so reconstructed instances are equally fast.

   Unflatten reverses the field split: dynamic children are restored (with any `_Static` wrappers in containers stripped), static fields are set directly, and the constructor is bypassed with `object.__new__` + `object.__setattr__` because constructors take different arguments than stored fields (`Linear(in_dim, out_dim, key)` creates `w` and `b` internally). This is also why we use `register_pytree_with_keys` instead of `register_dataclass`.

   The result is that `jax.jit` and `jax.grad` work natively with models without special wrappers.

3. **Freeze after init.** `__init__` is wrapped to set `_frozen` once construction completes. Subsequent attribute assignment raises `AttributeError`, because mutation would silently break JAX tracing. Use `model.at.field.set(new_value)` to create a modified copy: `at` returns a path-recording proxy, and `set` rebuilds the modules and containers along the recorded path (sharing all untouched subtrees) via the same `object.__new__` + `object.__setattr__` mechanism as unflatten.

## Optimizer (`ion/optimizer.py`)

Wraps an optax `GradientTransformation` with Param-aware updates. `Optimizer` is a standalone pytree registered directly via `register_pytree_node_class`, so it works with `jax.jit` and `jax.lax.scan`.

**Auto-partitioning.** On construction, if the model contains any bare JAX arrays or frozen `Param` leaves, the optimizer automatically wraps the transform with `optax.partition`, assigning the real optimizer to trainable params and `optax.set_to_zero()` to frozen params. This means no momentum/variance buffers are allocated for frozen weights. Important for LoRA and other fine-tuning setups where most params are frozen.

**Update.** `optimizer.update(model, grads)` calls the underlying optax transform, then applies the resulting deltas to the model's trainable `Param` leaves only. Non-`Param` arrays (like batch statistics) and frozen `Param` leaves pass through unchanged. The `Param` wrapper is preserved on updated values so trainability metadata survives the step.

**Step counter.** `optimizer.step` is an `int32` array that increments on each `update()` call. This is independent of any internal step tracking by optax transforms (e.g. warmup schedules).

**Per-field transforms.** Pass a dict instead of a single transform to route different optimizers to different top-level model fields. This is useful when different components need independent learning rates or gradient clipping, such as GANs (generator vs discriminator), actor-critic RL (separate LR and grad norm thresholds), or transfer learning (slow backbone, fast head).

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

Internally this uses `optax.partition` with labels derived from `jax.tree.map_with_path`. Every top-level field with trainable params must be covered by the dict; missing fields raise `ValueError` at construction time. Frozen params within any group are still routed to `set_to_zero()`.

**Pytree registration.** `state` and `step` are dynamic children (traced by JAX); `_transform` and `_fields` are static auxiliary data (baked into the compiled program).

## Precision

Layer constructors take no `dtype` argument. Parameters are created in JAX's default float dtype (`float32`, or `float64` when `jax_enable_x64` is set), and precision is controlled entirely through `astype`. There are three patterns:

- **Default (float32).** Build and train with no dtype handling.
- **Mixed precision.** Keep float32 master params and cast the model to `bfloat16` *inside* the loss with `ion.astype(model, jnp.bfloat16)`. The cast is differentiable, so gradients return in float32 to match the master params and the optimizer state; only the forward/backward math runs in bfloat16. This mirrors Keras `mixed_bfloat16`, PyTorch AMP, and Flax's `param_dtype`/`dtype` split. See [`examples/gpt_tinystories.ipynb`](../examples/gpt_tinystories.ipynb) for a worked example.
- **Full bfloat16 inference.** Cast once after construction with `model.astype(jnp.bfloat16)`.

## 🔪 Sharp Edges

Known gotchas when using Ion. Some are JAX limitations, others follow from Ion's design.

### Python scalars are compile-time constants

Plain Python scalars (ints, floats, strings) stored as module fields go into the treedef as static auxiliary data. JAX cannot trace them: they are invisible to `jax.grad` and fixed at `jax.jit` compile time. If a value needs to change at runtime (a temperature, a step counter), store it as a `jnp.array` or `Param`.

```python
# Static: recompiles if temperature changes
self.temperature = 0.5

# Dynamic: traced by JAX, no recompilation
self.temperature = jnp.array(0.5)
```

Every distinct set of static values compiles a separate trace, so changing one triggers recompilation. `Param.trainable` is static too: set trainability once, before training. Calling `freeze()`/`unfreeze()` inside a training loop recompiles every step.

### Pytrees cannot share references

JAX pytrees are trees, not graphs. If two fields point to the same `Module` or `Param`, JAX silently duplicates the object during flatten/unflatten, and updates to one copy stop affecting the other. For weight tying (e.g. shared embedding and output projection), reference the underlying array instead of storing the module twice:

```python
# Don't: the two fields become independent copies
self.embed = Embedding(vocab, dim, key=key)
self.output_proj = self.embed

# Do: reference the weight explicitly
self.embed = Embedding(vocab, dim, key=key)
# In __call__:
logits = x @ self.embed.w.T
```

### `save`/`load` doesn't store callables or static config

Non-array fields (ints, strings, activation functions) come from the reference tree passed to `load`, not from the file. If you change an activation in code and load an old checkpoint, you get the new activation with the old weights, with no warning. Array data and `trainable` flags do round-trip exactly, including `bfloat16`/`float8` (stored as raw bytes with the dtype in metadata); shape mismatches raise `ValueError`.

### `at` can change pytree structure

Setting a `Param` field to a plain array or `None` changes the treedef. That is useful for model surgery, but a later `jax.tree.map` between the original and modified model crashes with a structure mismatch.

### Optimizer state is bound to the model at construction

`ion.Optimizer` snapshots the model's pytree structure when created, including each `Param`'s `trainable` flag, and allocates no state for frozen params. If you then change the model, whether by `freeze()`/`unfreeze()` or by restructuring with `at`, the next `update()` raises `ValueError: Model structure or trainability changed, create a new Optimizer.` The fix is one line: rebuild the optimizer from the updated model.

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)
model = model.unfreeze()
optimizer.update(model, grads)  # ValueError

# Fix: rebuild the optimizer after changing the model
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

Rebuilding resets momentum buffers, which is what you want: newly unfrozen params have no gradient history. One case slips past the check: swapping a `Param` for one with a different shape leaves the treedef unchanged, so the mismatch surfaces as a shape error inside optax rather than Ion's `ValueError`. The same fix applies.

### Some lower-level LAX functions don't accept `Param` directly

Most `jnp` operations accept `Param` transparently. Lower-level functions like `lax.conv_general_dilated` require plain arrays: convert with `jnp.asarray(param)`, which goes through `__jax_array__` and applies `stop_gradient` for frozen params, so autograd correctness is preserved. **Never use `param._value` for this.** It bypasses `stop_gradient`, so frozen params receive real gradients during the backward pass, breaking the guarantee that frozen params produce zero gradients. The field is private, reserved for internal code that deliberately needs the raw array.

### A `bfloat16` model with `float32` inputs promotes back to `float32`

Casting the model alone buys you nothing: under JAX type promotion, `bfloat16` weights applied to `float32` inputs silently upcast every result back to `float32`. Cast both the model and its inputs.

```python
model = model.astype(jnp.bfloat16)
y = model(x.astype(jnp.bfloat16))
```

### Module immutability is shallow

`_frozen` prevents field reassignment, but mutable containers (lists, dicts, numpy arrays) stored in fields can still be mutated in place: `model.layers.append(...)` bypasses the freeze. Worse, mutating a static field in place does **not** trigger JIT recompilation, because JAX identifies pytree aux data by object identity, so the mutated list still hits the stale cached trace with the old value baked in. Use `at` to create a new module with the updated field.

### `Param.__eq__` returns a JAX array, not a bool

`param in list` can raise `ValueError` for multi-element params, because Python calls `bool()` on the array result, and truthiness is ambiguous for arrays with more than one element.

### Nested `jax.grad(jax.grad(f))` on `Param` raises `ValueError`

Differentiating with respect to a `Param` (or a `Module`, whose leaves are `Param`s) works fine with a single `jax.grad`, but nesting two fails:

```python
p = Param(jnp.array(2.0))
jax.grad(f)(p)            # works, gradient comes back as a Param
jax.grad(jax.grad(f))(p)  # ValueError
jax.hessian(f)(p)         # works, use this for second derivatives
```

The inner `grad` returns its gradient wrapped as a `Param`, and abstractifying that intermediate `Param` in the outer trace triggers `__jax_array__`, which JAX no longer supports during abstractification. In practice this rarely matters: nested `grad` is only valid for scalar-to-scalar functions anyway, even with plain arrays (the inner `grad` of `f: R^n -> R` returns a vector, which the outer `grad` rejects).

### `Module.params` preserves static fields

`model.params` replaces plain array leaves with `None`, while non-array fields (ints, floats, strings, callables) remain unchanged. This is by design: static fields are structural metadata stored in the treedef, not pytree leaves, so they are naturally unaffected when leaves are replaced.
