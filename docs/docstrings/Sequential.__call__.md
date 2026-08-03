Apply each layer in order.

Parameters
----------
x : Any
    Input to the first layer.
training : bool | None, default=None
    Training mode forwarded to layers that accept it.
key : jax.Array | None, default=None
    RNG key split across the contained layers.

Returns
-------
Any
    Output of the final layer.

Info
----
Arguments are forwarded according to each layer's signature. A layer with a
training mode requires an explicit value for `training`.
