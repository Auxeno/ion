Apply dropout to the input.

Parameters
----------
x : Float[Array, "..."]
    Input of any shape.
deterministic : bool | None, default=None
    Overrides the layer's `deterministic` flag for this call. `None` uses the
    value set at construction.
key : PRNGKeyArray | None
    RNG key for the dropout mask. Keyword-only. Required unless the call is
    deterministic.

Returns
-------
Float[Array, "..."]
    Masked and rescaled output (or `x` unchanged when deterministic), same
    shape as the input.
