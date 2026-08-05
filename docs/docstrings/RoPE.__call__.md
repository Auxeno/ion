Rotate features by position.

Parameters
----------
x : jax.Array["... d", float]
    Query or key with the per-head dimension last, and the sequence on the layer's
    `axis`. The head dimension must be divisible by `2 * len(shape)`, or by 2 when
    no `shape` is set. The sequence length must equal the number of lattice
    positions plus `num_prefix_tokens`.

Returns
-------
jax.Array["... d", float]
    Position-rotated array, same shape as the input.
