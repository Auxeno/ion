Rotate features by position.

Parameters
----------
x : jax.Array["... d", float]
    Query or key with the per-head dimension last. It must be divisible by
    `2 * len(shape)`, or by 2 when no `shape` is set.
axis : int, default=-2
    Axis containing the sequence positions. Its length must equal the number of
    lattice positions plus `num_prefix_tokens`.

Returns
-------
jax.Array["... d", float]
    Position-rotated array, same shape as the input.
