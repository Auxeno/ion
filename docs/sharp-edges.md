# 🔪 Sharp edges

Known gotchas when using Ion. Some are JAX limitations, others follow from Ion's design.

## What Ion leaves out

Some things are left out deliberately. There are no custom transforms and no training loop abstraction. The only mutable state is a declared `Buffer` field on the layer that owns it, as used by `BatchNorm`. Ion defines and trains models; JAX does everything else.

## Python scalars are compile-time constants

Plain Python scalars (ints, floats, strings) stored as module fields go into the treedef as static auxiliary data. JAX cannot trace them: they are invisible to `jax.grad` and fixed at `jax.jit` compile time. If a value needs to change at runtime (a temperature, a step counter), store it as a `jnp.array` or `Param`.

```python
# Static: recompiles if temperature changes
self.temperature = 0.5

# Dynamic: traced by JAX, no recompilation
self.temperature = jnp.array(0.5)
```

Every distinct set of static values compiles a separate trace, so changing one triggers recompilation. `Param.trainable` is static too: set trainability once, before training. Calling `freeze()`/`unfreeze()` inside a training loop recompiles every step.

## Models with buffers are not plain values

A [`Buffer`](core/buffers.md) is mutable, so a model owning one breaks the value semantics that hold everywhere else in Ion. A copy made with `jax.tree.map` shares its buffers with the original, and updating one updates both.

```python
copy = jax.tree.map(lambda leaf: leaf, model)  # shares running statistics
independent = model.clone()                    # owns running statistics
```

`clone`, `freeze`, `unfreeze` and `load` all return models with their own buffers. `astype` deliberately does not: the mixed-precision workflow casts inside the loss, and a cast copy with its own buffers would quietly throw away every update made through it. `Optimizer.update` shares them for the same reason, since a step continues one model rather than copying it.

### Buffer mutation and JAX transforms

`jax.vmap` may read a buffer, so stateful models work normally in evaluation. It cannot write one shared buffer from several mapped lanes: those concurrent writes have no defined ordering or final value. Ion buffers are unmapped pytree metadata, so a training call that writes one under `vmap` raises. Pass the whole batch to the layer instead; `BatchNorm` reduces over all leading axes. If an update must combine mapped values, compute them with `vmap`, reduce them, then call `set` once outside it. Use `jax.lax.scan` when updates must be sequential. The same restriction applies to `jax.shard_map` over a shared buffer and to transforms implemented with batching, such as `jax.jacfwd`.

`nn.Ensemble` uses `vmap` for member construction as well as evaluation, so its factory cannot return a model with `Buffer` fields. Creating a buffer inside the mapped factory lets a traced reference escape the transform; `BatchNorm`, `SpectralNorm`, and other stateful layers therefore cannot be ensemble members.

`jax.checkpoint` (also called `remat`) is unsafe around buffer writes for a different reason: it may replay the forward computation during the backward pass, applying the update a second time. Keep stateful operations outside the rematerialized function:

```python
x = norm(x, training=True)
x = jax.checkpoint(expensive_stateless_block)(x)
```

Independent buffers also have independent pytree structures because reference identity is part of a buffer's static metadata. Two separately constructed, cloned, or loaded stateful models therefore cannot be inputs to the same multi-tree `jax.tree.map`, and each owns a separate JIT specialization. Operate on `model.params` when combining or comparing parameter trees.

## Pytrees cannot share references

JAX [pytrees](https://docs.jax.dev/en/latest/pytrees.html) are trees, not graphs. If two fields point to the same `Module` or `Param`, JAX silently duplicates the object during flatten/unflatten, and updates to one copy stop affecting the other. For weight tying (e.g. shared embedding and output projection), reference the underlying array instead of storing the module twice:

```python
# Don't: the two fields become independent copies
self.embed = Embedding(vocab, dim, key=key)
self.output_proj = self.embed

# Do: reference the weight explicitly
self.embed = Embedding(vocab, dim, key=key)
# In __call__:
logits = x @ self.embed.w.T
```

## `save`/`load` doesn't store callables or static config

Non-array fields (ints, strings, activation functions) come from the reference tree passed to `load`, not from the file. If you change an activation in code and load an old checkpoint, you get the new activation with the old weights, with no warning. Array data and `trainable` flags do round-trip exactly, including `bfloat16`, `float8` and `complex64`; shape mismatches raise `ValueError`.

## `at` can change pytree structure

Setting a `Param` field to a plain array or `None` changes the treedef. That is useful for model surgery, but a later `jax.tree.map` between the original and modified model crashes with a structure mismatch.

## `at` type steps match outermost instances only

A type step stops descending at each match, so instances nested inside another match are not selected: `model.at[Sequential]` on a Sequential that contains another Sequential only touches the outer one. Each match also receives the same `value` object, so array values are shared across matches, not re-initialized per match.

## Optimizer state is bound to the model at construction

`ion.Optimizer` snapshots the model's pytree structure when created, including each `Param`'s `trainable` flag, and allocates no state for frozen params. If you then change the model, whether by `freeze()`/`unfreeze()` or by restructuring with `at`, the next `update()` raises `ValueError: Model structure or trainability changed, create a new Optimizer.` The fix is one line: rebuild the optimizer from the updated model.

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)
model = model.unfreeze()
optimizer.update(model, grads)  # ValueError

# Fix: rebuild the optimizer after changing the model
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

Rebuilding resets momentum buffers, which is what you want: newly unfrozen params have no gradient history. One case slips past the check: swapping a `Param` for one with a different shape leaves the treedef unchanged, so the mismatch surfaces as a shape error inside optax rather than Ion's `ValueError`. The same fix applies.

## Some lower-level LAX functions don't accept `Param` directly

Most `jnp` operations accept `Param` transparently. Lower-level functions like `lax.conv_general_dilated` require plain arrays: use `param.value`, which applies `stop_gradient` for frozen params, so autograd correctness is preserved. **Never use `param._value` for this.** It is the raw stored array and bypasses `stop_gradient`, so frozen params receive real gradients during the backward pass, breaking the guarantee that frozen params produce zero gradients. It is reserved for internal code that deliberately needs raw storage, such as checkpointing and dtype casting.

## A `bfloat16` model with `float32` inputs promotes back to `float32`

Casting the model alone buys you nothing: under JAX type promotion, `bfloat16` weights applied to `float32` inputs silently upcast every result back to `float32`. Cast both the model and its inputs.

```python
model = model.astype(jnp.bfloat16)
y = model(x.astype(jnp.bfloat16))
```

## Numerically sensitive reductions use `float32`

Normalization layers, `AvgPool`, and floating-point segment reductions compute in `float32`, even when JAX's 64-bit mode is enabled, and cast results back to the input dtype. Write a custom operation if a reduction itself must use `float64`.

## Module immutability is shallow

`_frozen` prevents field reassignment, but mutable containers (lists, dicts, numpy arrays) stored in fields can still be mutated in place: `model.layers.append(...)` bypasses the freeze. Worse, mutating a static field in place does **not** trigger JIT recompilation, because JAX identifies pytree aux data by object identity, so the mutated list still hits the stale cached trace with the old value baked in. Use `at` to create a new module with the updated field.

## `Param.__eq__` returns a JAX array, not a bool

`param in list` can raise `ValueError` for multi-element params, because Python calls `bool()` on the array result, and truthiness is ambiguous for arrays with more than one element.

## Nested `jax.grad(jax.grad(f))` on `Param` raises `ValueError`

Differentiating with respect to a `Param` (or a `Module`, whose leaves are `Param`s) works fine with a single `jax.grad`, but nesting two fails:

```python
p = Param(jnp.array(2.0))
jax.grad(f)(p)            # works, gradient comes back as a Param
jax.grad(jax.grad(f))(p)  # ValueError
jax.hessian(f)(p)         # works, use this for second derivatives
```

The inner `grad` returns its gradient wrapped as a `Param`, and abstractifying that intermediate `Param` in the outer trace triggers `__jax_array__`, which JAX no longer supports during abstractification. In practice this rarely matters: nested `grad` is only valid for scalar-to-scalar functions anyway, even with plain arrays (the inner `grad` of \(f\colon \mathbb{R}^n \to \mathbb{R}\) returns a vector, which the outer `grad` rejects).

## `Module.params` preserves static fields

`model.params` replaces plain array and `Buffer` fields with `None`, while non-array fields (ints, floats, strings, callables) remain unchanged. This is by design: static fields are structural metadata stored in the treedef, not pytree leaves, so they are naturally unaffected when dynamic values are replaced.

## `ion.cost` only breaks down what it scopes

The layer table comes from scopes `cost` opens around each `__call__`, so work running outside one is folded into the whole-call total. The totals stay correct; only the per-layer attribution is lost, and the report prints a single row.

A transform rebuilds the model as it traces, and the rebuilt copy reclaims its layer paths from whichever call entered it. That call is `__call__`, so a loss or training step written against the forward pass breaks down normally. A model entered through some other method has nothing to reclaim from:

```python
ion.cost(jax.grad(loss), model, x)      # loss calls model(x): full breakdown
ion.cost(train_step, model, opt, x, y)  # loss calls model.actor(x): one row
```

Analyse that branch on its own with `model.cost(x, method="actor")` and read the gradient total from the transform report. The same limit applies one level down: a layer that transforms its own submodules, such as a `lax.scan` over stacked blocks, rebuilds them where no scope reaches, so they do not appear.

## `jax.checkpoint` under `jax.vmap` reports per-lane shapes

`ion.cost` records each layer's output shape as the layer traces, restoring the axis `vmap` maps over. A `jax.checkpoint` region traces its body before that axis is added, so layers inside one report a single lane while layers outside keep the mapped axis, and a report carries both:

```python
class Net(nn.Module):
    def __call__(self, x):
        return self.head(jax.checkpoint(self.block)(x))

ion.cost(lambda m, x: jax.vmap(m)(x), net, jnp.ones((8, 16)))
# block  f32(16,)   one lane
# head   f32(8, 4)  all eight
```

Only the shape column is affected: FLOPs, memory and operation counts describe the whole mapped call throughout. Passing the batch to the model directly, rather than mapping over single examples, avoids it.
