# Internals

Everything behind the scenes in Ion. Four files and ~400 lines of code — that's the whole engine. This document explains the design, but readers are encouraged to check out the source code as it's fairly straightforward:

- [`ion/nn/module.py`](../ion/nn/module.py) — Module base class, pytree registration
- [`ion/nn/param.py`](../ion/nn/param.py) — Param wrapper, trainable/frozen distinction
- [`ion/tree.py`](../ion/tree.py) — Static wrapper, apply_updates, save/load
- [`ion/transforms.py`](../ion/transforms.py) — grad/value_and_grad for trainable params only

## Module (`ion/nn/module.py`)

JAX requires two things from objects in `jit`/`grad`/`vmap`: pytree registration so JAX can traverse their structure, and immutability so tracing produces correct results. Plain Python classes satisfy neither.

Three things happen in `__init_subclass__` when a class inherits from `Module`:

1. **Dataclass conversion** — `@dataclasses.dataclass` is applied. If the
   subclass defines its own `__init__`, it is kept; otherwise one is generated
   from the annotations.

2. **Pytree registration** — registered with `register_pytree_with_keys`. During flatten, non-array leaves (ints, floats, bools, strings, callables) are wrapped in `Static` so JAX treats them as compile-time constants rather than traced values. `Module` and `Param` fields are returned as-is since they have their own pytree registrations. During unflatten, `Static` wrappers are stripped so the user sees plain values. This means `jax.jit` and `jax.grad` work natively with models — no special wrappers needed. We use `register_pytree_with_keys` instead of `register_dataclass` because constructors take different arguments than stored fields (`Linear(in_dim, out_dim, key)` creates `w` and `b` internally). Unflattening bypasses the constructor with `object.__new__` + `object.__setattr__`.

3. **Freeze after init** — `__init__` is wrapped to set `_frozen` once construction completes. Subsequent attribute assignment raises `AttributeError`, because mutation would silently break JAX tracing. Use `model.replace(field=new_value)` to create a modified copy.

## Param (`ion/nn/param.py`)

JAX pytrees see all arrays equally — there is no built-in way to distinguish trainable weights from frozen weights from plain buffers. `Param` makes this explicit:

- `Param(array)` — trainable (`ion.grad` differentiates, optimizers update)
- `Param(array, trainable=False)` — frozen (carried through but excluded from gradients)
- bare array — plain data buffer, never treated as a parameter

### Pytree registration

`Param` is registered via `register_dataclass` with `value` as a dynamic child (traced/differentiated by JAX) and `trainable` as static metadata (baked into compiled programs as a cache key). Changing `trainable` triggers recompilation, but it's a one-time flag set at construction.

### Transparent array behavior

`__jax_array__` lets `jnp.*` unwrap automatically (`jnp.dot(param, x)` works). `__getattr__` forwards `.shape`, `.dtype`, etc. to the inner array. Arithmetic and comparisons return raw arrays, not `Param`, because intermediate results are not parameters.

## Tree (`ion/tree.py`)

### Static

A pytree node with no children — JAX treats its value as static metadata. Used by `Module` pytree registration to wrap non-array leaves (ints, strings, callables) so they become invisible to JAX tracing while being preserved through flatten/unflatten roundtrips.

### apply_updates

Adds optimizer deltas to a model's trainable parameters. Walks the model and update trees in parallel, skipping positions where the update is `None` or the parameter is frozen (`Param(trainable=False)`). The `Param` wrapper is preserved on updated values so trainability metadata survives the step.

### save / load

`save` flattens the tree and writes only array leaves to a `.npz` file, keyed by position index.

`load` reads them back into a reference tree that supplies the original structure — non-array leaves are left untouched, so static config (layer sizes, activation functions) comes from the reference model, not the file.

## Transforms (`ion/transforms.py`)

### grad / value_and_grad

`jax.grad` differentiates every float array in its first argument — there is no way to exclude frozen parameters or non-parameter arrays like batch statistics.

`ion.grad` solves this in three steps:

1. Flatten the first arg, split leaves into trainable `Param`s vs everything else
2. Pass only trainable leaves to `jax.grad`, holding everything else constant
3. Pad non-trainable positions with `None` and unflatten back to the original pytree shape

The caller gets a gradient tree matching the model structure, where only trainable `Param` positions have values and everything else is `None`. `value_and_grad` works identically but also returns the output.

Note: `jax.jit` works natively with all modules because `Module` pytree registration wraps non-array leaves in `Static` automatically. No `ion.jit` wrapper is needed.

## Sharp Edges

Known gotchas to be aware of when using Ion. Some are limitations of JAX:

- **`save`/`load` only stores array data** — `trainable` flags and non-array fields (ints, strings, callables) come from the reference tree, not the file. If you save a frozen model and load into a trainable reference, the loaded model will be trainable.

- **Module immutability is shallow** — `_frozen` prevents field reassignment, but mutable containers (lists, dicts, numpy arrays) in fields can still be mutated in-place. For example, `model.layers.append(...)` bypasses the freeze.

- **`Param.__eq__` returns a JAX array, not a bool** — `param in list` can raise `ValueError` for multi-element params because Python calls `bool()` on the array result, which is ambiguous for arrays with more than one element.

- **`Module.params` preserves static fields alongside `Param` leaves** — plain arrays become `None`, while non-array fields (ints, floats, strings, callables) remain unchanged. This is by design: static fields are structural metadata stored in the treedef, not pytree leaves, so they are naturally unaffected when `params` replaces non-`Param` leaves with `None`.
