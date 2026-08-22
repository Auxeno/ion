Analyse one call's arithmetic and memory, layer by layer.

Parameters
----------
*args : Any
    Arguments for the call. A `jax.ShapeDtypeStruct` stands in for an array.
method : str, default='__call__'
    Name of the method to analyse.
**kwargs : Any
    Keyword arguments for the call. Non-array configuration compiles in as static.

Returns
-------
Cost
    Call totals and a `LayerCost` for every layer the trace named, keyed by tree path.

Example
-------
```python
model = nn.MLP([256, 512, 10], key=key)
report = model.cost(jnp.ones((32, 256)))

print(report)
```
