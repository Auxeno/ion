Describe the static work and memory implied by one call, layer by layer.

The call is traced and compiled without being executed. Concrete array inputs are replaced
by shape/dtype placeholders, so an array and the equivalent `jax.ShapeDtypeStruct` produce
the same analysis. Module scopes attribute traced operations and outputs to the layer that
created them.

The target may be a model, in which case the remaining arguments are passed to it, or any
function taking a model among its arguments. That covers a loss, a gradient, or a whole
training step, including both the forward and reverse work in a gradient evaluation.

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
model = nn.MLP([256, 512, 10], key=key)
print(ion.cost(model, jnp.ones((32, 256))))

report = ion.cost(jax.grad(loss), model, x, y)
report.layers["layers[0]"].share  # its fraction of the call's FLOPs
```
