Add the transformed input to the original input.

Parameters
----------
x : jax.Array["...", float]
    Input to both branches.
*args : Any
    Additional positional arguments forwarded to the wrapped layer.
training : bool | None, default=None
    Training mode forwarded when the wrapped layer accepts it.
key : jax.Array | None, default=None
    RNG key forwarded when the wrapped layer accepts it.
**kwargs : Any
    Additional keyword arguments forwarded to the wrapped layer.

Returns
-------
jax.Array["...", float]
    Elementwise sum of the input and transformed input.
