Apply batch normalization.

Parameters
----------
x : jax.Array["... d", float]
    Input with features last.
buffers : Buffers
    Buffers returned by `model.init_buffers()`.
training : bool
    Whether to use batch statistics and update the running values.

Returns
-------
tuple[jax.Array["... d", float], Buffers]
    Normalized output and updated buffers.

Info
----
Keep the returned buffers after every training call.
