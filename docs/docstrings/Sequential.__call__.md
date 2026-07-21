Apply each layer in order.

Parameters
----------
x : jax.Array
    Input to the first layer.
key : jax.Array | None
    Optional RNG key. Keyword-only. When given, it is split and forwarded to
    any layers whose signature accepts a `key` (such as `Dropout`).

Returns
-------
jax.Array
    Output of the final layer.
