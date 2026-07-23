Apply dropout to the input.

Parameters
----------
x : jax.Array["...", float]
    Input of any shape.
deterministic : bool | None, default=None
    Overrides the layer's `deterministic` flag for this call. `None` uses the
    value set at construction.
key : jax.Array | None
    RNG key for the dropout mask. Keyword-only. Required unless the call is
    deterministic.

Returns
-------
jax.Array["...", float]
    Masked and rescaled output (or `x` unchanged when deterministic), same
    shape as the input.

Info
----
Pass a `key` at call time to sample the mask; omit it (or set
`deterministic=True`) to pass the input through unchanged. A `key` is required
unless the call is deterministic.
