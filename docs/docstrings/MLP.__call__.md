Apply the layers in sequence.

Parameters
----------
x : Float[Array, "... i"]
    Input with feature dimension `dims[0]` last. Any number of leading batch
    dimensions is supported.

Returns
-------
Float[Array, "... o"]
    Output with feature dimension `dims[-1]`.
