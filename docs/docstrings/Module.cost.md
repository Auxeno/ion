Analyse one call's arithmetic and memory, layer by layer.

The call is traced and compiled without being executed. Concrete array inputs are replaced
by shape/dtype placeholders, so an array and the equivalent `jax.ShapeDtypeStruct` produce
the same analysis. Module scopes attribute traced operations and outputs to the layer that
created them. A layer that transforms its own submodules rebuilds them as it traces, so they
are not named and do not appear in the report.

Parameters
----------
*args : Any
    Arguments for the call. A `jax.ShapeDtypeStruct` stands in for an array.
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
