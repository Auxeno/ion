Apply each layer in order.

Parameters
----------
x : Any
    Input to the first layer.
buffers : Buffers | None, default=None
    Model buffers. When provided, updated buffers are returned with the output.
training : bool | None, default=None
    Training mode forwarded to layers that accept it.
key : jax.Array | None, default=None
    RNG key split across the contained layers.

Returns
-------
Any | tuple[Any, Buffers]
    Output of the final layer, and updated buffers when `buffers` is provided.

Info
----
Arguments are forwarded according to each layer's signature. A stateful layer
requires `buffers`, and a layer with a training mode requires an explicit value
for `training`.
