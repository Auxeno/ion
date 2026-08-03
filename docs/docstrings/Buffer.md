Marks a JAX array as a non-trainable value updated in place.

Parameters
----------
value : jax.Array
    Initial value. Read it back with `buffer.value`.

Example
-------
```python
class MeanCenter(nn.Module):
    running_mean: nn.Buffer
    momentum: float

    def __init__(self, dim, momentum=0.1):
        self.running_mean = nn.Buffer(jnp.zeros(dim))
        self.momentum = momentum

    def __call__(self, x, *, training):
        if training:
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=axes)
            self.running_mean.set(
                (1.0 - self.momentum) * self.running_mean.value
                + self.momentum * mean
            )
        else:
            mean = self.running_mean.value

        return x - mean


layer = MeanCenter(64)
y = layer(x, training=True)
```

Note
----
Buffers hold state such as BatchNorm running statistics. They live in the model
but contribute no pytree leaves, so `jax.grad`, `ion.Optimizer` and
`Module.astype` leave them alone.

A buffer is mutable, so a model owning one is not a plain value: a copy made
with `jax.tree.map` shares its state, as does the model returned by
`Module.astype`. `Module.clone`, `Module.freeze` and `Module.unfreeze` all give
the model they return its own buffers.
