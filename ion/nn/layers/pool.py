"""Pooling layers.

Modules:
    MaxPool  Downsample by taking the max over windows.
    AvgPool  Downsample by averaging over windows.

Channels-last format: (..., spatial, channels).
Stride defaults to kernel size when not specified.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..module import Module


class MaxPool(Module):
    """N-dimensional max pooling.

    >>> pool = MaxPool(2, kernel_size=2)
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    num_spatial_dims: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[tuple[int, int], ...]

    def __init__(
        self,
        num_spatial_dims: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: str | int | tuple[int, ...] = 0,
    ) -> None:

        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims ({num_spatial_dims}) must be >= 1")

        self.num_spatial_dims = num_spatial_dims

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * num_spatial_dims
        self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride,) * num_spatial_dims
        else:
            self.stride = stride

        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        else:
            self.padding = tuple((p, p) for p in padding)

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        num_spatial = self.num_spatial_dims
        batch_shape = x.shape[: -(num_spatial + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial + 1) :])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        x = lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, *self.kernel_size, 1),
            window_strides=(1, *self.stride, 1),
            padding=padding,
        )

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial + 1) :])

        return x


class AvgPool(Module):
    """N-dimensional average pooling.

    >>> pool = AvgPool(2, kernel_size=2)
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    num_spatial_dims: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[tuple[int, int], ...]

    def __init__(
        self,
        num_spatial_dims: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: str | int | tuple[int, ...] = 0,
    ) -> None:

        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims ({num_spatial_dims}) must be >= 1")

        self.num_spatial_dims = num_spatial_dims

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * num_spatial_dims
        self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride,) * num_spatial_dims
        else:
            self.stride = stride

        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        else:
            self.padding = tuple((p, p) for p in padding)

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        num_spatial = self.num_spatial_dims
        batch_shape = x.shape[: -(num_spatial + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial + 1) :])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        window_dims = (1, *self.kernel_size, 1)
        window_strides = (1, *self.stride, 1)

        ones = jnp.ones_like(x)

        x = lax.reduce_window(
            operand=x,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_dims,
            window_strides=window_strides,
            padding=padding,
        )

        window_counts = lax.reduce_window(
            operand=ones,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_dims,
            window_strides=window_strides,
            padding=padding,
        )

        x = x / window_counts

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial + 1) :])

        return x
