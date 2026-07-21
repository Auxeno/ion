# Module

The base class for everything in Ion. Subclassing `Module` turns a plain class into an immutable JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html), so `jax.jit`, `jax.grad`, and `jax.vmap` work on your models directly.

::: ion.nn.Module
    options:
      members:
        - at
        - freeze
        - unfreeze
        - astype
        - params
        - num_params

## How it works

Three things happen in `__init_subclass__` when a class inherits from `Module`:

1. **Dataclass conversion.** `@dataclasses.dataclass` is applied. If the subclass defines its own `__init__`, it is kept; otherwise one is generated from the annotations.

2. **Pytree registration.** The class is registered with `register_pytree_with_keys`. Each field is classified once at construction time (via `isinstance` checks) and the result is cached on the instance:

   - **Array-like** (`Param`, `Module`, `jax.Array`, `np.ndarray`) become dynamic children, passed to JAX as-is.
   - **Containers with array-like content at any depth** (a tuple of `Module`s in `Sequential`, a `list[list[Module]]`) become dynamic children. Pure containers traverse natively; mixed containers have their non-array elements wrapped in `_Static` at any nesting depth so JAX treats them as compile-time constants.
   - **Everything else** (int, float, str, callable, None, containers with no arrays anywhere) becomes static auxiliary data, stored in the treedef directly.

   Since modules are frozen after `__init__`, the classification never changes and subsequent flatten calls skip the `isinstance` checks entirely. Unflatten bypasses the constructor with `object.__new__` + `object.__setattr__`, because constructors take different arguments than stored fields (`Linear(in_dim, out_dim, key)` creates `w` and `b` internally). This is why registration uses `register_pytree_with_keys` rather than `register_dataclass`.

3. **Freeze after init.** `__init__` is wrapped to set `_frozen` once construction completes. Subsequent attribute assignment raises `AttributeError`, because mutation would silently break JAX tracing. Use `at` to create a modified copy: it returns a path-recording proxy, and `set` rebuilds the modules along the recorded path (sharing all untouched subtrees). A type as a step fans out to every matching node: `model.at[nn.Dropout].p.set(0.5)`. Zero matches raise `ValueError`.

For the design rationale and the immutability gotchas, see [Sharp Edges](../guides/sharp-edges.md).
