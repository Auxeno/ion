Normalize the last dimension.

Parameters
----------
x : jax.Array["... d", float]
    Input with feature dimension `dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
jax.Array["... d", float]
    Normalized and affine-transformed output, same shape as the input.

Info
----
Operates on the last dimension only, so any number of leading batch dimensions
is supported.
