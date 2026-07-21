Normalize the last dimension.

Parameters
----------
x : Float[Array, "... d"]
    Input with feature dimension `dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
Float[Array, "... d"]
    Normalized and affine-transformed output, same shape as the input.
