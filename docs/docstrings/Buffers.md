Collection of non-trainable model values updated during forward passes.

Note
----
Create buffers with `Module.init_buffers` and pass them through each forward
call. A `BufferModule` reads its value with `buffers[self]` and updates it with
`buffers.set(self, value)`.

Example
-------
```python
buffers = model.init_buffers()
y, buffers = model(x, buffers, training=True)
```
