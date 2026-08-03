Activate treescope as the default interactive renderer.

Parameters
----------
everything : bool, optional
    If `True`, render all types rather than only Ion types and arrays. Default
    `False`.

Example
-------
```python
ion.enable_treescope()                 # Ion types and arrays only
ion.enable_treescope(everything=True)  # all types
```
