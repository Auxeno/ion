Apply the frozen base layer plus the scaled low-rank update.

Parameters
----------
x : Float[Array, "... i"]
    Input with feature dimension `in_dim` last. Any number of leading batch
    dimensions is supported.

Returns
-------
Float[Array, "... o"]
    `linear(x) + (x @ A @ B) * (alpha / rank)`, projected to `out_dim`.
