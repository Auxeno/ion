# Module

`Module` gives an Ion model its structure. Modules contain arrays, params, buffers, configuration, and other modules, forming a JAX [pytree](https://docs.jax.dev/en/latest/pytrees.html).

::: ion.nn.Module
    options:
      members:
        - at
        - clone
        - freeze
        - unfreeze
        - astype
        - params
        - num_params
        - disk_usage

---

## Fields

Declare fields with class annotations and assign them in `__init__`:

| Value | Use it for |
|---|---|
| [`Param`](param.md) | Model parameters. |
| [`Buffer`](buffers.md) | Non-trainable array state mutated during a forward pass. |
| JAX array | Other array data. |
| `Module` | Child layers. |
| Python value | Static configuration. |

```python
class Block(nn.Module):
    linear: nn.Linear
    activation: Callable

    def __init__(self, dim, activation=jax.nn.relu, *, key):
        self.linear = nn.Linear(dim, dim, key=key)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.linear(x))
```

## Immutability

Fields are frozen after `__init__`. Model edits return a new module:

```python
model = model.at.encoder.layers[0].set(new_layer)
model = model.freeze()
model = model.astype(jnp.bfloat16)
```

[`Buffer`](buffers.md) contents are the exception: a module cannot replace the field, but may update the array inside it during a forward pass.

## How does it work?

Modules, params, and arrays become dynamic pytree children. Python values such as integers, strings, callables, and `None` become static metadata. Ion reconstructs transformed modules without rerunning their constructors, so models work directly with `jax.jit`, `jax.grad`, and `jax.vmap`.
