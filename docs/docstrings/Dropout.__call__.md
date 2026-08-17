Apply dropout to the input.

Parameters
----------
x : jax.Array["...", float]
    Input of any shape.
training : bool
    Whether to apply dropout. Evaluation returns the input unchanged.
key : jax.Array | None, default=None
    RNG key for the dropout mask. Required during training when `p > 0`.

Returns
-------
jax.Array["...", float]
    Masked and rescaled output, or `x` unchanged during evaluation. The shape
    and dtype match the input.

Info
----
`training` is explicit because modules do not store a mutable training mode.
