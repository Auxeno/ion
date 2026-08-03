# Module

The base class for everything in Ion. Subclassing `Module` turns a plain class
into an immutable JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html), so
models work directly with `jax.jit`, `jax.grad`, and `jax.vmap`.

::: ion.nn.Module
    options:
      members:
        - at
        - init_buffers
        - freeze
        - unfreeze
        - astype
        - params
        - num_params

---

## Immutability

Modules are frozen after `__init__`. Direct assignment raises
`AttributeError`; methods that change a module return a new value instead.

```python
model = model.at.encoder.layers[0].set(new_layer)
model = model.freeze()
model = model.astype(jnp.bfloat16)
```

Untouched subtrees are shared with the original model. Changing pytree
structure or trainability after constructing an optimizer requires a new
`Optimizer`. See [Sharp edges](../sharp-edges.md).

Stateful layers keep their non-trainable values in a separate
[`Buffers`](buffers.md) collection rather than mutating module fields.

## How does it work?

Fields containing a `Param`, `Module`, array, or a container of those values
become dynamic pytree children. JAX traces, differentiates, and transforms
them. Configuration values such as integers, strings, callables, and `None`
become static metadata stored in the pytree structure.

```python
class Block(nn.Module):
    linear: nn.Linear     # dynamic
    activation: Callable  # static
    width: int            # static
```

Static values are compile-time constants, so changing one creates a new JIT
specialization. Field classification happens once after construction and is
then fixed by module immutability.

Ion registers each subclass as a keyed JAX pytree and reconstructs transformed
modules without rerunning their constructors. This allows constructors to
create stored parameters from dimensions and an RNG key while JAX transforms
operate directly on the resulting fields.
