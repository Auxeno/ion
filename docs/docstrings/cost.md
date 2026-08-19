Analyse one call's arithmetic and memory, layer by layer, for any function taking a model.

[`Module.cost`](core/module.md#ion.nn.Module.cost) covers a model's own forward pass. This
function also accepts a callable, so a loss, a gradient, or a whole training step is
analysed the same way, including the reverse work in a gradient evaluation.

Parameters
----------
target : Module | Callable
    A model to call, or a function taking a model among its arguments.
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
report = ion.cost(jax.grad(loss), model, x, y)
print(report)
```
