Attend over the sequence, or into a context sequence.

Parameters
----------
x : jax.Array["b s d", float]
    Query sequence of `s` tokens with feature dimension `dim`.
x_kv : jax.Array["b t c", float] | None, default=None
    Context sequence of `t` tokens with feature dimension `kv_dim`, supplying
    the keys and values. `None` draws them from `x`, giving self-attention.
mask : jax.Array["...", bool] | None, default=None
    Optional boolean mask over query-key pairs, where `True` means attend and
    `False` means ignore. Accepts `(s, t)`, `(b, s, t)`, or `(b, h, s, t)`.
    Composes with `causal` and `window`. Fully masked query heads contribute
    zeros.

Returns
-------
jax.Array["b s d", float]
    Attention output over the query sequence.
