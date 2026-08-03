# Buffers

Some layers, such as `BatchNorm`, update running values during a forward pass.
These values live outside the immutable model in a `Buffers` collection.

::: ion.nn.Buffers
    options:
      show_docstring_description: false
      members:
        - set

::: ion.nn.BufferModule
    options:
      members: false

---

## Using buffers

Users typically only refer to `Buffers` itself in type annotations. Create
buffers with `model.init_buffers()` and pass the returned collection between
calls:

```python
model = nn.Sequential(
    nn.Linear(3, 64, key=key),
    nn.BatchNorm(64),
    jax.nn.relu,
)
buffers = model.init_buffers()

y, buffers = model(x, buffers, training=True)
y, _ = model(x, buffers, training=False)
```

Evaluation reads the latest values without changing them. `SpectralNorm` needs
a random key when its buffers are initialized:

```python
buffers = model.init_buffers(key=key)
```

`Sequential` forwards buffers to stateful layers and returns the final
collection. In a custom `Module`, pass the latest collection to each stateful
layer explicitly.

## Writing a stateful layer

Subclass `BufferModule` when a layer owns one non-trainable buffer value.
Implement `_init_buffer`, read the value with `buffers[self]`, and replace it
with `buffers.set(self, value)`:

```python
class MeanCenter(nn.BufferModule):
    dim: int
    momentum: float

    def __init__(self, dim, momentum=0.1):
        self.dim = dim
        self.momentum = momentum

    def _init_buffer(self, *, key=None):
        running_mean = jnp.zeros(self.dim)
        return running_mean

    def __call__(self, x, buffers, *, training):
        running_mean = buffers[self]

        if training:
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=axes)
            running_mean = (
                (1.0 - self.momentum) * running_mean
                + self.momentum * mean
            )
            buffers = buffers.set(self, running_mean)
        else:
            mean = running_mean

        return x - mean, buffers
```

Buffer reads and stored updates automatically apply `jax.lax.stop_gradient`,
so existing state is treated as constant by autodiff. A newly computed value
reused in the same call remains differentiable; stop it explicitly if that
calculation should not affect parameter gradients.

A buffer value must be a non-empty pytree of arrays. Its pytree structure,
leaf shapes, and leaf dtypes are fixed at initialization so one collection has
a stable structure under JAX transformations.

Buffer precision is owned by the layer and chosen in `_init_buffer`. Existing
buffers are separate from the model and are not cast by `model.astype`. Users
should not need to cast buffers themselves. A layer can keep numerically
sensitive state in float32, but that internal dtype should not unexpectedly
promote its output. Convert values back to the dtype the layer would normally
return at the boundary of the buffered calculation.

## Model identity

Buffers are associated with their owning `BufferModule` instances. Parameter
updates and model transformations preserve that identity, so the same buffers
continue to work with the updated model. Replacing a stateful layer creates a
new identity and requires `model.init_buffers()` again.

Optimizers only update model parameters, while buffer updates remain explicit
in the forward pass. See [Stateful training](../workflows.md#stateful-training)
for a complete training step.
