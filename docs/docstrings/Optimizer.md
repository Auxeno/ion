Wraps an optax transformation with `Param`-aware, pytree-native updates.

Parameters
----------
tx : optax.GradientTransformation or dict
    A single optax transform, or a dictionary mapping top-level model field
    names (or tuples of names) to per-field transforms.
model : PyTree
    Model used to initialize state and partition frozen parameters and bare
    arrays out of the transform.

Attributes
----------
step : jax.Array
    `int32` update counter.
state : PyTree
    Internal optax optimizer state.

Example
-------
```python
learning_rate = 3e-4

optimizer = ion.Optimizer(optax.adam(learning_rate), model)
grads = jax.grad(loss_fn)(model, x, y)
model, optimizer = optimizer.update(model, grads)
```
