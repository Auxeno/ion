Normalize over channel groups and the trailing spatial dimensions.

Parameters
----------
x : jax.Array["... d", float]
    Channels-last input with `num_spatial_dims` trailing spatial dimensions
    before the channel dimension `dim`.

Returns
-------
jax.Array["... d", float]
    Normalized and affine-transformed output, same shape as the input.

Info
----
Channels-last format.
