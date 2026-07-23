Attend from the query sequence into the context.

Parameters
----------
x : jax.Array["b s d", float]
    Query sequence of `s` tokens with feature dimension `dim`.
context : jax.Array["b t c", float]
    Context (key/value) sequence of `t` tokens with feature dimension
    `context_dim`.
mask : jax.Array["...", bool] | None, default=None
    Optional boolean mask over query-key pairs, where `True` means attend and
    `False` means ignore. Accepts `(s, t)`, `(b, s, t)`, or `(b, h, s, t)`.

Returns
-------
jax.Array["b s d", float]
    Attention output over the query sequence.