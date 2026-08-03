Initialize the model's non-trainable buffer values.

Parameters
----------
key : jax.Array | None, default=None
    Random key for buffer initialization. The key is split once per unique
    `BufferModule`.

Returns
-------
Buffers
    Buffer values keyed by their owning modules.

Example
-------
```python
buffers = model.init_buffers()
y, buffers = model(x, buffers, training=True)
```
