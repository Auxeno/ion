Holds persistent, non-trainable array state mutated by a forward pass.

Parameters
----------
value : jax.Array
    Initial value. Read it back with `buffer.value`.

Example
-------
```python
running_mean = nn.Buffer(jnp.zeros(16))
current_mean = running_mean.value

running_mean.set(jnp.ones(16))
updated_mean = running_mean.value
```

Note
----
Use a plain JAX array unless the value must be mutated during a forward pass.
Buffers have a fixed shape and dtype, contribute no pytree leaves, and apply
`jax.lax.stop_gradient` on writes.
