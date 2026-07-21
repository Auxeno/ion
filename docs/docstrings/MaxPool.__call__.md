Apply max pooling.

Parameters
----------
x : Float[Array, "b *spatial c"]
    Input in channels-last format with exactly one leading batch dimension
    and `len(kernel_shape)` spatial dimensions.

Returns
-------
Float[Array, "b *spatial c"]
    Pooled output. Output spatial dimensions depend on stride and padding.
