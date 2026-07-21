Attend over the sequence.

Parameters
----------
x : jax.Array["b s d", float]
    Input sequence of `s` tokens with feature dimension `dim`.
mask : jax.Array["...", bool] | None, default=None
    Optional boolean mask where `True` means attend and `False` means ignore.
    Accepts `(s, s)`, `(b, s, s)`, or `(b, h, s, s)`. Composes with `causal`
    and `window`.

Returns
-------
jax.Array["b s d", float]
    Attention output, same shape as the input.
