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

Notes
-----
Channels-last format: input shape is `(batch, *spatial, channels)` with exactly one leading batch dimension. Use `jax.vmap` for extra batch dimensions.

Examples
--------
>>> conv_t = nn.ConvTranspose(3, 16, kernel_shape=(5,), padding=2, key=key)          # 1D
>>> conv_t = nn.ConvTranspose(3, 16, kernel_shape=(3, 3), padding="VALID", key=key)   # 2D
>>> y = conv_t(x)  # (b, h, w, 3) -> (b, h', w', 16)
