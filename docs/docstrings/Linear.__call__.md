Apply the layer to the last axis of `x`.

Parameters
----------
x : jax.Array["... i", float]
    Input with feature dimension `in_dim` last.

Returns
-------
jax.Array["... o", float]
    Input projected to `out_dim` features, leading dimensions unchanged.

Info
----
Applies to the last dimension, so any number of leading batch dimensions is supported.
