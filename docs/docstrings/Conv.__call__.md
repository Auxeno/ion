Apply the convolution.

Parameters
----------
x : Float[Array, "b *spatial c"]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
Float[Array, "b *spatial c"]
    Convolved output. Output spatial dimensions depend on stride, padding,
    and dilation.
