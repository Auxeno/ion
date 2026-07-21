Apply the layers in sequence.

Parameters
----------
x : jax.Array["... i", float]
    Input with feature dimension `dims[0]` last. Any number of leading batch
    dimensions is supported.

Returns
-------
jax.Array["... o", float]
    Output with feature dimension `dims[-1]`.
