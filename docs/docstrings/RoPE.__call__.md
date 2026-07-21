Rotate features by position.

Parameters
----------
x : Float[Array, "... s d"]
    Query or key with sequence positions on the second-to-last axis and an
    even per-head dimension last.

Returns
-------
Float[Array, "... s d"]
    Position-rotated array, same shape as the input.
