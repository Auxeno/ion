Apply the transposed convolution.

Parameters
----------
x : jax.Array["b *spatial c", float]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
jax.Array["b *spatial c", float]
    Upsampled output. Output spatial dimensions depend on stride, padding,
    output_padding, and dilation.
