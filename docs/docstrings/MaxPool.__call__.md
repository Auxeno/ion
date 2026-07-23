Apply max pooling.

Parameters
----------
x : jax.Array["b *spatial c", float]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
jax.Array["b *spatial c", float]
    Pooled output. Output spatial dimensions depend on stride and padding.

Info
----
Channels-last format; use `jax.vmap` for extra leading batch dimensions.
