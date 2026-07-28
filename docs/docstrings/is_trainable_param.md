Check if an object is a trainable `Param`.

Parameters
----------
x : Any
    Object to test.

Returns
-------
bool
    `True` if `x` is a `Param` with `trainable=True`.

Example
-------
```python
ion.is_trainable_param(model.encoder.w)
```
