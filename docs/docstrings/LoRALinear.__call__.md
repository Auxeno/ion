Apply the frozen base layer plus the scaled low-rank update.

Parameters
----------
x : jax.Array["... i", float]
    Input with feature dimension `in_dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
jax.Array["... o", float]
    \(\operatorname{linear}(x) + xAB(\alpha/r)\), projected to `out_dim`.
