Path-based model surgery that returns modified copies.

Navigate through fields, indices, dictionary keys, or module types, then call
`.set(value)`. Untouched subtrees are shared with the original model.

Returns
-------
object
    Path-recording proxy rooted at this module.

Example
-------
```python
model = model.at.encoder.layers[0].set(new_layer)
model = model.at[nn.Dropout].p.set(0.0)
```
