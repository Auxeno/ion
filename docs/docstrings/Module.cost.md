Analyse one call's arithmetic and memory, layer by layer.

The call is traced and compiled without being executed. Concrete array inputs are replaced
by shape/dtype placeholders, so an array and the equivalent `jax.ShapeDtypeStruct` produce
the same analysis. Module scopes attribute traced operations and outputs to the layer that
created them.

Parameters
----------
*args : Any
    Arguments for the call. A `jax.ShapeDtypeStruct` stands in for an array.
**kwargs : Any
    Keyword arguments for the call. Non-array configuration compiles in as static.

Returns
-------
Cost
    Call totals and a `LayerCost` for every module, keyed by tree path.

Example
-------
```python
model = nn.MLP([256, 512, 10], key=key)
report = model.cost(jnp.ones((32, 256)))

print(report)
```
