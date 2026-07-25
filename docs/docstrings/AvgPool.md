N-dimensional average pooling.

Averages over each sliding window. Spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, and 3D.

Parameters
----------
kernel_shape : tuple[int, ...]
    Window dimensions. The tuple length sets the spatial rank.
stride : int | tuple[int, ...] | None, default=None
    Step between windows. Defaults to `kernel_shape` (non-overlapping windows).
    A scalar broadcasts across all spatial dimensions.
padding : Literal["SAME", "VALID"] | int | tuple[int, ...], default=0
    `"SAME"`, `"VALID"`, or numeric padding applied to both sides of each
    spatial dimension. A scalar broadcasts; a tuple gives per-dimension control.

Example
-------
```python
batch, height, width, channels = 8, 32, 32, 3
pool = nn.AvgPool(kernel_shape=(2, 2))
x = jnp.ones((batch, height, width, channels))
y = pool(x)  # (8, 32, 32, 3) -> (8, 16, 16, 3)

x_batched = jnp.ones((5, batch, height, width, channels))  # extra batch dim
y_batched = jax.vmap(pool)(x_batched)  # (5, 8, 32, 32, 3) -> (5, 8, 16, 16, 3)
```
