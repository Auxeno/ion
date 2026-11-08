Load array leaves and metadata from a `.ion` file into a reference pytree.

Parameters
----------
path : str | os.PathLike
    Path to a `.ion` file created by `save` (`.ion` appended if missing).
reference_pytree : PyTree
    Provides the tree structure. Its array and Python scalar leaves are replaced.

Returns
-------
PyTree
    Pytree with arrays, scalars and `Param` trainable flags restored from file.

Example
-------
```python
model = ion.load("model.ion", model)
```
