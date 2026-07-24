Return a copy with every parameter trainable.

Returns
-------
Module
    Module with all `Param` leaves set to `trainable=True`.

Example
-------
```python
model = model.unfreeze()
```
