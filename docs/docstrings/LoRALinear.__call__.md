Apply the frozen base layer plus the scaled low-rank update.

Parameters
----------
x : jax.Array["... i", float]
    Input with feature dimension `in_dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
jax.Array["... o", float]
    `linear(x) + (x @ A @ B) * (alpha / rank)`, projected to `out_dim`.
