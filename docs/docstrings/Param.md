Marks a JAX array as a trainable or frozen model parameter.

Parameters
----------
_value : jax.Array
    Array to wrap. Access it through `jnp.asarray(param)`, not the private
    `_value` field.
trainable : bool, default=True
    Whether gradients flow through the parameter. Frozen parameters apply
    `jax.lax.stop_gradient` and receive no optimizer state.

Attributes
----------
trainable : bool
    Whether the parameter is trainable.

Example
-------
```python
in_dim, out_dim = 3, 16

w = nn.Param(jnp.zeros((in_dim, out_dim)))
b = nn.Param(jnp.zeros(out_dim), trainable=False)

w.shape  # (3, 16)
b.trainable  # False
```
