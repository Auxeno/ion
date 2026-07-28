Load array leaves and metadata from a `.ion` file into a reference pytree.

Parameters
----------
path : str
    Path to a `.ion` file created by `save` (`.ion` appended if missing).
reference_pytree : PyTree
    Provides tree structure and non-array leaves; array leaves are replaced.

Returns
-------
PyTree
    Pytree with arrays and `Param` trainable flags restored from file.

Example
-------
```python
model = ion.load("model.ion", model)
```
