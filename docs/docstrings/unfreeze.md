Return a copy with all `Param`s set to `trainable=True`.

Parameters
----------
pytree : PyTree
    Pytree containing `Param` wrappers, plain arrays, or both.

Returns
-------
PyTree
    Copy of the pytree with every `Param` unfrozen.

Example
-------
```python
unfrozen_model = ion.unfreeze(model)
```
