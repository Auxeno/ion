Serialize a pytree's array leaves and metadata to a `.ion` file.

Parameters
----------
path : str | os.PathLike
    Destination file path (`.ion` appended if missing).
pytree : PyTree
    Pytree to serialize. Array leaves, Python scalar leaves and `Param` trainable flags
    are written.

Example
-------
```python
ion.save("model.ion", model)
```
