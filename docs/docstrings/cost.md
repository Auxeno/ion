Analyse one call's arithmetic and memory, layer by layer, for any function taking a model.

The call is traced and compiled without being executed. Concrete array inputs are replaced
by shape/dtype placeholders, so an array and the equivalent `jax.ShapeDtypeStruct` produce
the same analysis. Module scopes attribute traced operations and outputs to the layer that
created them. A layer that transforms its own submodules rebuilds them as it traces, so they
are not named and do not appear in the report.

[`Module.cost`](core/module.md#ion.nn.Module.cost) covers a model's own forward pass. This
function also accepts a callable, so a loss, a gradient, or a whole training step is
analysed the same way, including the reverse work in a gradient evaluation, which is charged
to the layer whose forward pass produced it. A bound method such as `model.critic` names the
model and the call together.

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
    Call totals and a `LayerCost` for every layer the trace named, keyed by tree path.

Example
-------
```python
ion.cost(jax.grad(loss), model, x, y)
ion.cost(model.critic, x)
```
