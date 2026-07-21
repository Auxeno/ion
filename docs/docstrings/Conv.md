N-dimensional convolution layer.

Spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, 3D, and beyond.

Parameters
----------
in_channels : int
    Input channels.
out_channels : int
    Output channels.
kernel_shape : tuple[int, ...]
    Spatial kernel dimensions. The tuple length sets the spatial rank.
stride : int | tuple[int, ...], default=1
    Convolution stride. A scalar broadcasts across all spatial dimensions.
padding : str | int | tuple[int, ...], default=0
    `"SAME"`, `"VALID"`, or numeric padding applied to both sides of each
    spatial dimension. A scalar broadcasts; a tuple gives per-dimension control.
dilation : int | tuple[int, ...], default=1
    Dilation rate. A scalar broadcasts across all spatial dimensions.
groups : int, default=1
    Number of groups for grouped convolution. Both `in_channels` and
    `out_channels` must be divisible by `groups`.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. He normal by default, suited to ReLU networks.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Kernel of shape `(*kernel_shape, in_channels // groups, out_channels)`.
b : Param | None
    Bias vector of shape `(out_channels,)`. `None` when `bias=False`.

Notes
-----
Channels-last format: input shape is `(batch, *spatial, channels)` with exactly
one leading batch dimension. Use `jax.vmap` for extra batch dimensions.

Examples
--------
>>> conv1d = nn.Conv(3, 16, kernel_shape=(5,), key=key)                # 1D
>>> conv2d = nn.Conv(3, 16, kernel_shape=(3, 3), padding=1, key=key)   # 2D
>>> y = conv2d(x)  # (b, h, w, 3) -> (b, h', w', 16)
