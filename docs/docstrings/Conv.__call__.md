Apply the convolution.

Parameters
----------
x : jax.Array["b *spatial c", float]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
jax.Array["b *spatial c", float]
    Convolved output. Output spatial dimensions depend on stride, padding,
    and dilation.
