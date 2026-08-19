Measure what one call costs, layer by layer.

The call is traced and compiled once. Arithmetic is counted from the jaxpr, where loops are
still visible, and memory traffic from the compiled HLO, where fusion is. Both passes read
the scopes each `Module.__call__` installs while tracing, so every operation is attributed
to the layer that produced it rather than guessed at from dataflow.

The target may be a model, in which case the remaining arguments are passed to it, or any
function taking a model among its arguments. That covers a loss, a gradient, or a whole
training step, and a backward pass is measured as readily as a forward one.

Parameters
----------
target : Module | Callable
    A model to call, or a function to call with the arguments that follow.
*args : Any
    Arguments for the call. A `jax.ShapeDtypeStruct` stands in for an array.
balance : float, default=250.0
    Peak arithmetic per peak byte for the device in mind, the ridge dividing compute
    limited work from bandwidth limited work. Only `share` and `ceiling` depend on it.
**kwargs : Any
    Keyword arguments for the call. Those holding no array compile in as static.

Returns
-------
Cost
    Totals for the call and a `LayerCost` for every module, keyed by tree path.

Example
-------
```python
model = nn.MLP([256, 512, 10], key=key)
print(ion.cost(model, jnp.ones((32, 256))))

report = ion.cost(jax.grad(loss), model, x, y)  # a training step
report.layers["layers[0]"].ceiling              # 0.05, so bandwidth limited
```
