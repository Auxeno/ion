Activate treescope as the default interactive renderer.

Parameters
----------
everything : bool, optional
    If `True`, render all types rather than only Ion modules and params. Default `False`.

Example
-------
```python
ion.enable_treescope()                 # Ion modules and params only
ion.enable_treescope(everything=True)  # all types
```
