Return a copy with matching array leaves cast to a dtype.

Parameters
----------
dtype : jnp.dtype
    Target dtype. Only leaves in the same dtype family are cast.
params_only : bool, default=False
    If `True`, cast only `Param` leaves. Keyword-only.

Returns
-------
Module
    Cast copy of the module.

Example
-------
```python
model = model.astype(jnp.bfloat16)
```
