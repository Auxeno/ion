Return a copy with all `Param`s set to `trainable=False`.

Parameters
----------
pytree : PyTree
    Pytree containing `Param` wrappers, plain arrays, or both.

Returns
-------
PyTree
    Copy of the pytree with every `Param` frozen.

Example
-------
```python
frozen_model = ion.freeze(model)
```
