# Internals

Everything behind the scenes in Ion. Three files and ~250 lines of code make up the whole engine. This document explains the design. Readers are also encouraged to check out the source code as it's fairly straightforward:

- [`ion/nn/module.py`](../ion/nn/module.py): Module base class, pytree registration
- [`ion/nn/param.py`](../ion/nn/param.py): Param wrapper, trainable/frozen distinction
- [`ion/tree.py`](../ion/tree.py): Static wrapper, apply_updates, freeze/unfreeze

## Module (`ion/nn/module.py`)

JAX requires two things from objects in `jit`/`grad`/`vmap`: pytree registration so JAX can traverse their structure, and immutability so tracing produces correct results. Plain Python classes satisfy neither.

Three things happen in `__init_subclass__` when a class inherits from `Module`:

1. **Dataclass conversion.** `@dataclasses.dataclass` is applied. If the
   subclass defines its own `__init__`, it is kept; otherwise one is generated
   from the annotations.

2. **Pytree registration.** The class is registered with `register_pytree_with_keys`. During flatten, non-array leaves (ints, floats, bools, strings, callables) are wrapped in `Static` so JAX treats them as compile-time constants rather than traced values. `Module` and `Param` fields are returned as-is since they have their own pytree registrations. During unflatten, `Static` wrappers are stripped so the user sees plain values. This means `jax.jit` and `jax.grad` work natively with models without special wrappers. We use `register_pytree_with_keys` instead of `register_dataclass` because constructors take different arguments than stored fields (`Linear(in_dim, out_dim, key)` creates `w` and `b` internally). Unflattening bypasses the constructor with `object.__new__` + `object.__setattr__`.

3. **Freeze after init.** `__init__` is wrapped to set `_frozen` once construction completes. Subsequent attribute assignment raises `AttributeError`, because mutation would silently break JAX tracing. Use `model.replace(field=new_value)` to create a modified copy.

## Param (`ion/nn/param.py`)

JAX pytrees see all arrays equally and have no built-in way to distinguish trainable weights from frozen weights from plain buffers. `Param` makes this explicit:

- `Param(array)`: trainable (gradients flow normally, optimizers update)
- `Param(array, trainable=False)`: frozen (`stop_gradient` applied via `__jax_array__`, making this parameter invisible to autodiff)
- bare array: plain data buffer, never treated as a parameter

### Pytree registration

`Param` is registered via `register_dataclass` with `_value` as a dynamic child (traced/differentiated by JAX) and `trainable` as static metadata (baked into compiled programs as a cache key). Changing `trainable` triggers recompilation, but it's a one-time flag set at construction.

### Transparent array behavior

`__jax_array__` returns the raw array for trainable params and applies `jax.lax.stop_gradient` for frozen params, making the `trainable` flag physically real in JAX's autodiff. `__getattr__` routes attribute access (`.shape`, `.dtype`, `.T`, `.reshape(...)`) through `jnp.asarray(self)`, which calls `__jax_array__()`, so frozen params remain invisible to autodiff even through method calls. The `_value` field is private and should not be accessed directly in user code, as it bypasses `stop_gradient`. Arithmetic and comparisons return raw arrays, not `Param`, because intermediate results are not parameters.

## Tree (`ion/tree.py`)

### Static

A pytree node with no children. JAX treats its value as static metadata. Used by `Module` pytree registration to wrap non-array leaves (ints, strings, callables) so they become invisible to JAX tracing while being preserved through flatten/unflatten roundtrips.

### apply_updates

Adds optimizer deltas to a model's trainable parameters. Only `Param` leaves are modified; non-`Param` arrays (like batch statistics) pass through unchanged. Walks the model and update trees in parallel, skipping positions where the update is `None`, the leaf is not a `Param`, or the parameter is frozen (`Param(trainable=False)`). The `Param` wrapper is preserved on updated values so trainability metadata survives the step.

## 🔪 Sharp Edges

Known gotchas to be aware of when using Ion. Some are limitations of JAX:

- **Python ints, floats, and strings are static, not dynamic.** When stored as module fields, plain Python scalars are wrapped in `Static` and baked into the compiled program. JAX cannot trace through them, so they are invisible to `jax.grad` and fixed at `jax.jit` compile time. Changing a static value (including `Param.trainable`) triggers JIT recompilation since every unique combination compiles a separate trace. If you need a value to be dynamic at runtime (e.g. a temperature parameter, a step counter), store it as a `jnp.array` or `Param` instead. Similarly, avoid calling `freeze()`/`unfreeze()` inside a training loop as it recompiles on every step. Set trainability once and keep it fixed.

  ```python
  # Static: recompiles if temperature changes
  self.temperature = 0.5

  # Dynamic: traced by JAX, no recompilation
  self.temperature = jnp.array(0.5)
  ```

- **Pytrees cannot share references to the same object.** JAX pytrees are trees, not graphs. If two fields point to the same `Module` or `Param`, JAX duplicates the object during flatten/unflatten and updates to one copy won't affect the other. For weight tying (e.g. shared embedding and output projection), reference the underlying array directly instead of storing the same module twice:

  ```python
  # Don't do this, the two fields become independent copies
  self.embed = Embedding(vocab, dim, key=key)
  self.output_proj = self.embed  # silent duplication

  # Do this instead, reference the weight explicitly
  self.embed = Embedding(vocab, dim, key=key)
  # In __call__:
  logits = x @ self.embed.w.T  # shared weight, no duplication
  ```

- **`save`/`load` doesn't store callables or static config.** Non-array fields (ints, strings, callables like activation functions) come from the reference tree, not the file. Array data and `trainable` flags are saved and restored. Shape mismatches between saved and reference arrays raise `ValueError`.

- **`replace()` can change pytree structure.** Replacing a `Param` field with a plain array or `None` changes the treedef. This is useful for model surgery, but subsequent `jax.tree.map` between the original and modified model will crash with a structure mismatch.

- **`apply_updates` only modifies `Param` leaves.** Non-`Param` arrays are left unchanged. Frozen `Param` updates are silently skipped. Note that `jax.grad` does compute gradients for all array leaves in the pytree, including non-`Param` arrays. The optimizer will track state for these leaves, but `apply_updates` discards their updates. Continuing to calculate non-parameter array gradients could may be desirable to some users so Ion leaves it in.

- **Some JAX/LAX functions don't accept `Param` directly.** Most operations will work transparently because `Param.__jax_array__` converts automatically, but lower-level functions like `lax.conv_general_dilated` may reject a `Param` where they expect a plain array. Use `jnp.asarray(param)` to convert. This calls `__jax_array__`, which applies `stop_gradient` for frozen params, so autograd correctness is preserved. **Do not use `param._value`**, which bypasses `stop_gradient` entirely: frozen params would receive gradients during the backward pass (wasting compute), and only `apply_updates` silently discarding them would prevent actual weight changes. The `_value` field is private and reserved for internal code that deliberately needs the raw array.

- **Module immutability is shallow.** `_frozen` prevents field reassignment, but mutable containers (lists, dicts, numpy arrays) in fields can still be mutated in-place. For example, `model.layers.append(...)` bypasses the freeze. Worse, in-place mutation of a `Static`-wrapped field (like a list of ints) will **not** trigger JIT recompilation because JAX identifies the pytree aux data by object identity, so the same mutated list still hits the stale cached trace with the old value baked in. Use `replace()` to create a new module with the updated field instead.

- **`Param.__eq__` returns a JAX array, not a bool.** `param in list` can raise `ValueError` for multi-element params because Python calls `bool()` on the array result, which is ambiguous for arrays with more than one element.

- **Nested `jax.grad(jax.grad(f))` on `Param` raises `ValueError`.** JAX triggers `__jax_array__()` during abstractification of the intermediate gradient `Param`, which is no longer supported. In practice this rarely matters: `jax.grad(jax.grad(f))` only works on scalar-to-scalar functions even with plain arrays (the inner `grad` of `f: R^n -> R` returns a vector, which the outer `grad` rejects). Use `jax.hessian` for second-order derivatives, which works correctly with both `Param` and `Module`.

- **`Module.params` preserves static fields alongside `Param` leaves.** Plain arrays become `None`, while non-array fields (ints, floats, strings, callables) remain unchanged. This is by design: static fields are structural metadata stored in the treedef, not pytree leaves, so they are naturally unaffected when `params` replaces non-`Param` leaves with `None`.
