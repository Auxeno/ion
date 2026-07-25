Pass-through module that returns its first argument unchanged and ignores the
rest.

Keeps a disabled module slot visible and replaceable through `Module.at`.

Parameters
----------
*args : Any
    Positional arguments accepted and ignored.
**kwargs : Any
    Keyword arguments accepted and ignored.

Example
-------
```python
model = nn.Sequential(nn.LayerNorm(64))
model = model.at.layers[0].set(nn.Identity())
```
