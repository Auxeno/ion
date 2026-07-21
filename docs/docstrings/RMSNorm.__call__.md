Normalize the last dimension by its root mean square.

Parameters
----------
x : Float[Array, "... d"]
    Input with feature dimension `dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
Float[Array, "... d"]
    Rescaled output, same shape as the input.
