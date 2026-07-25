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
padding : Literal["SAME", "VALID"] | int | tuple[int, ...], default=0
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
    Weight initializer. Glorot uniform by default.
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

Example
-------
```python
batch, length, in_channels, out_channels = 4, 10, 3, 16
conv1d = nn.Conv(in_channels, out_channels, kernel_shape=(5,), key=key)
x = jnp.ones((batch, length, in_channels))
y = conv1d(x)  # (4, 10, 3) -> (4, 6, 16)

batch, height, width = 8, 32, 32
conv2d = nn.Conv(in_channels, out_channels, kernel_shape=(3, 3), padding=1, key=key)
x = jnp.ones((batch, height, width, in_channels))
y = conv2d(x)  # (8, 32, 32, 3) -> (8, 32, 32, 16)

x = jnp.ones((5, batch, height, width, in_channels))  # extra batch dim
y = jax.vmap(conv2d)(x)  # (5, 8, 32, 32, 3) -> (5, 8, 32, 32, 16)
```
