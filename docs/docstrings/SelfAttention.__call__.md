Attend over the sequence.

Parameters
----------
x : Float[Array, "b s d"]
    Input sequence of `s` tokens with feature dimension `dim`.
mask : Bool[Array, ...] | None, default=None
    Optional boolean mask where `True` means attend and `False` means ignore.
    Accepts `(s, s)`, `(b, s, s)`, or `(b, h, s, s)`. Composes with `causal`
    and `window`.

Returns
-------
Float[Array, "b s d"]
    Attention output, same shape as the input.
