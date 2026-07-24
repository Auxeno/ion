Return a copy with every parameter frozen.

Returns
-------
Module
    Module with all `Param` leaves set to `trainable=False`.

Example
-------
```python
model = model.freeze()
```
