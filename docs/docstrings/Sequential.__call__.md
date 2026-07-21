Apply each layer in order.

Parameters
----------
x : Array
    Input to the first layer.
key : PRNGKeyArray | None
    Optional RNG key. Keyword-only. When given, it is split and forwarded to
    any layers whose signature accepts a `key` (such as `Dropout`).

Returns
-------
Array
    Output of the final layer.
