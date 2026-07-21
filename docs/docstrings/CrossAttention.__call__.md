Attend from the query sequence into the context.

Parameters
----------
x : Float[Array, "b s d"]
    Query sequence of `s` tokens with feature dimension `dim`.
context : Float[Array, "b t c"]
    Context (key/value) sequence of `t` tokens with feature dimension
    `context_dim`.
mask : Bool[Array, ...] | None, default=None
    Optional boolean mask over query-key pairs. Accepts `(s, t)`, `(b, s, t)`,
    or `(b, h, s, t)`.

Returns
-------
Float[Array, "b s d"]
    Attention output over the query sequence.
