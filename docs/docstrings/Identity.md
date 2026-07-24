Pass-through module that returns its first argument unchanged and ignores the
rest.

Keeps a disabled module slot visible and replaceable through `Module.at`.

Example
-------
```python
model = nn.Sequential(nn.LayerNorm(64))
model = model.at.layers[0].set(nn.Identity())
```
