Apply the convolution.

Parameters
----------
x : jax.Array["b ... i", float]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
jax.Array["b ... o", float]
    Convolved output. Output spatial dimensions depend on stride, padding,
    and dilation.

Info
----
Channels-last format: input shape is `(batch, *spatial, channels)` with exactly
one leading batch dimension. Use `jax.vmap` for extra batch dimensions.
