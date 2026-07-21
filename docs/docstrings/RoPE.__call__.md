Rotate features by position.

Parameters
----------
x : jax.Array["... s d", float]
    Query or key with sequence positions on the second-to-last axis and an
    even per-head dimension last.

Returns
-------
jax.Array["... s d", float]
    Position-rotated array, same shape as the input.
