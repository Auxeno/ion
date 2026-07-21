N-dimensional max pooling.

Takes the maximum over each sliding window. Spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, and 3D.

Parameters
----------
kernel_shape : tuple[int, ...]
    Window dimensions. The tuple length sets the spatial rank.
stride : int | tuple[int, ...] | None, default=None
    Step between windows. Defaults to `kernel_shape` (non-overlapping windows).
    A scalar broadcasts across all spatial dimensions.
padding : str | int | tuple[int, ...], default=0
    `"SAME"`, `"VALID"`, or numeric padding applied to both sides of each
    spatial dimension. A scalar broadcasts; a tuple gives per-dimension control.

Notes
-----
Holds no parameters and takes no `key`. Channels-last format with exactly one leading batch dimension; use `jax.vmap` for extra batch dimensions.

Examples
--------
>>> pool = nn.MaxPool(kernel_shape=(2, 2))
>>> y = pool(x)  # (b, h, w, c) -> (b, h // 2, w // 2, c)
