Apply each layer in order.

Parameters
----------
x : Any
    Input to the first layer.
key : jax.Array | None
    Optional RNG key. Keyword-only.

Returns
-------
Any
    Output of the final layer.

Info
----
Pass a `key` at call time to drive any stochastic layers in the chain (such as `Dropout`); it is split and forwarded only to layers whose signature accepts one.
