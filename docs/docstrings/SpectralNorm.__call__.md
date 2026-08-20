Apply the wrapped module with its normalized parameter.

Parameters
----------
x : jax.Array["...", float]
    Input to the wrapped module.
training : bool
    Whether to update the power-iteration vectors.

Returns
-------
jax.Array["...", float]
    Wrapped module output.
