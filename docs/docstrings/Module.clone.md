Return a copy whose buffers are independent of this model's.

Returns
-------
Module
    Module with a new `Buffer` for each of the original's.

Example
-------
```python
evaluation_model = model.clone()
```
