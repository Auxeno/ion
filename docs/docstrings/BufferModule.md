Base class for layers with one non-trainable buffer value updated during
forward passes. BatchNorm running statistics are one example.

Implement `_init_buffer`, read the value with `buffers[self]`, and update it
with `buffers.set(self, value)`. Buffer values stay outside the immutable model
and are not updated by the optimizer.

Example
-------
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


layer = MeanCenter(64)
buffers = layer.init_buffers()
y, buffers = layer(x, buffers, training=True)
```

Note
----
A `BufferModule` cannot contain another `BufferModule`; use a regular `Module`
to compose stateful layers. A buffer value must be a non-empty pytree of arrays,
and its structure, shapes, and dtypes cannot change after initialization.
