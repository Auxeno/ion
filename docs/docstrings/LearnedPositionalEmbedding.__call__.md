Add positional embeddings to the input.

Parameters
----------
x : Float[Array, "... s d"]
    Input with sequence positions on the second-to-last axis (length `<= max_len`)
    and feature dimension `dim` last.

Returns
-------
Float[Array, "... s d"]
    Input with the first `s` positional rows added, same shape as the input.
