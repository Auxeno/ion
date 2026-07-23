Normalize the last dimension by its root mean square.

Parameters
----------
x : jax.Array["... d", float]
    Input with feature dimension `dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
jax.Array["... d", float]
    Rescaled output, same shape as the input.

Info
----
Operates on the last dimension only, so any number of leading batch dimensions
is supported.
