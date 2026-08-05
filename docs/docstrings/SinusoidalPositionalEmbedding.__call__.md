Add sinusoidal positional encodings to the input.

Parameters
----------
x : jax.Array["... s d", float]
    Input with sequence positions on the second-to-last axis and the feature
    dimension last. The feature dimension must be even, since sine and cosine
    alternate across it.

Returns
-------
jax.Array["... s d", float]
    Input with the encoding for positions `0` to `s - 1` added, same shape as
    the input.
