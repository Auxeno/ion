N-dimensional transposed convolution layer.

The gradient of a convolution with respect to its input, commonly used for learnable upsampling in decoders and generators. Spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, and 3D.

Parameters
----------
in_channels : int
    Input channels.
out_channels : int
    Output channels.
kernel_shape : tuple[int, ...]
    Spatial kernel dimensions. The tuple length sets the spatial rank.
stride : int | tuple[int, ...], default=1
    Stride of the equivalent forward convolution; the upsampling factor. A
    scalar broadcasts across all spatial dimensions.
padding : str | int | tuple[int, ...], default=0
    `"SAME"`, `"VALID"`, or numeric padding. A scalar broadcasts; a tuple gives
    per-dimension control.
output_padding : int | tuple[int, ...], default=0
    Extra size added to one side of each output dimension, to disambiguate the
    output shape when `stride > 1`.
dilation : int | tuple[int, ...], default=1
    Dilation rate. A scalar broadcasts across all spatial dimensions.
groups : int, default=1
    Number of groups for grouped convolution. Both `in_channels` and
    `out_channels` must be divisible by `groups`.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Kernel of shape `(*kernel_shape, out_channels // groups, in_channels)`.
b : Param | None
    Bias vector of shape `(out_channels,)`. `None` when `bias=False`.

Info
----
Channels-last format: input shape is `(batch, *spatial, channels)` with exactly one leading batch dimension. Use `jax.vmap` for extra batch dimensions.

Example
-------
```python
# 1D: 2x spatial upsampling (stride=2, kernel=4, padding=1)
conv_t1d = nn.ConvTranspose(3, 16, kernel_shape=(4,), stride=2, padding=1, key=key)  
x = jnp.ones((4, 10, 3))
y = conv_t1d(x)  # (4, 10, 3) -> (4, 20, 16)

# 2D: 2x spatial upsampling with output_padding to resolve shape ambiguity
conv_t2d = nn.ConvTranspose(3, 16, kernel_shape=(3, 3), stride=2, padding=1, output_padding=1, key=key)  
x = jnp.ones((8, 16, 16, 3))
y = conv_t2d(x)  # (8, 16, 16, 3) -> (8, 32, 32, 16)

x = jnp.ones((5, 8, 16, 16, 3))  # extra batch dim
y = jax.vmap(conv_t2d)(x)  # (5, 8, 16, 16, 3) -> (5, 8, 32, 32, 16)
```