# 🔪 Sharp edges

Known gotchas when using Ion. Some are JAX limitations, others follow from Ion's design.

## What Ion leaves out

Some things are left out deliberately. There is no hidden mutable module state,
no custom transforms, and no training loop abstraction. Stateful layers such as
`BatchNorm` use explicit buffer collections. Ion defines and trains models; JAX
does everything else.

## Python scalars are compile-time constants

Plain Python scalars (ints, floats, strings) stored as module fields go into the treedef as static auxiliary data. JAX cannot trace them: they are invisible to `jax.grad` and fixed at `jax.jit` compile time. If a value needs to change at runtime (a temperature, a step counter), store it as a `jnp.array` or `Param`.

```python
# Static: recompiles if temperature changes
self.temperature = 0.5

# Dynamic: traced by JAX, no recompilation
self.temperature = jnp.array(0.5)
```

Every distinct set of static values compiles a separate trace, so changing one triggers recompilation. `Param.trainable` is static too: set trainability once, before training. Calling `freeze()`/`unfreeze()` inside a training loop recompiles every step.

## Buffers belong to their model

Call `model.init_buffers()` once and keep the collection returned by each
training call. Initializing buffers every step resets state such as BatchNorm
running statistics.

Buffers are associated with the `BufferModule` instances that created them.
Parameter updates, `model.at` edits within an existing layer, and JAX
transformations preserve this identity. Replacing a stateful layer creates a new
identity, so initialize buffers again for the changed model.

Buffer pytree structure, leaf shapes, and leaf dtypes cannot change after
initialization. This keeps the structure stable under `jax.jit`; incompatible
updates raise an error at `buffers.set(...)`.

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

Most `jnp` operations accept `Param` transparently. Lower-level functions like `lax.conv_general_dilated` require plain arrays: convert with `jnp.asarray(param)`, which goes through `__jax_array__` and applies `stop_gradient` for frozen params, so autograd correctness is preserved. **Never use `param._value` for this.** It bypasses `stop_gradient`, so frozen params receive real gradients during the backward pass, breaking the guarantee that frozen params produce zero gradients. The field is private, reserved for internal code that deliberately needs the raw array.

## A `bfloat16` model with `float32` inputs promotes back to `float32`

Casting the model alone buys you nothing: under JAX type promotion, `bfloat16` weights applied to `float32` inputs silently upcast every result back to `float32`. Cast both the model and its inputs.

```python
model = model.astype(jnp.bfloat16)
y = model(x.astype(jnp.bfloat16))
```

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

`model.params` replaces plain array leaves with `None`, while non-array fields (ints, floats, strings, callables) remain unchanged. This is by design: static fields are structural metadata stored in the treedef, not pytree leaves, so they are naturally unaffected when leaves are replaced.
