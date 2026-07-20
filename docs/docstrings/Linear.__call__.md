Apply the layer to the last axis of `x`.

Parameters
----------
x : Float[Array, "... i"]
    Input with feature dimension `in_dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
Float[Array, "... o"]
    Input projected to `out_dim` features, leading dimensions unchanged.
