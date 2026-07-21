Normalize over channel groups and the trailing spatial dimensions.

Parameters
----------
x : Float[Array, "... d"]
    Channels-last input with `num_spatial_dims` trailing spatial dimensions
    before the channel dimension `dim`.

Returns
-------
Float[Array, "... d"]
    Normalized and affine-transformed output, same shape as the input.
