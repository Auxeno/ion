Serialize a pytree's array leaves and metadata to a `.ion` file.

Parameters
----------
path : str
    Destination file path (`.ion` appended if missing).
pytree : PyTree
    Pytree to serialize. Only array leaves and `Param` trainable flags are written.

Example
-------
```python
ion.save("model.ion", model)
```
