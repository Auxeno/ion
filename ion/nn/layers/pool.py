"""Pooling layers.

Modules:
    MaxPool  Downsample by taking the max over windows.
    AvgPool  Downsample by averaging over windows.

Channels-last format: (..., spatial, channels).
Stride defaults to kernel shape when not specified.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..module import Module


class MaxPool(Module):
    """N-dimensional max pooling.

    >>> pool = MaxPool(kernel_shape=(2, 2))
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    kernel_shape: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[tuple[int, int], ...]

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: str | int | tuple[int, ...] = 0,
    ) -> None:

        if isinstance(kernel_shape, int):
            raise TypeError(
                f"kernel_shape must be a tuple of ints (e.g. (2,) or (2, 2)), got int {kernel_shape}"
            )
        if len(kernel_shape) < 1:
            raise ValueError("kernel_shape must have at least one element")

        self.kernel_shape = kernel_shape

        num_spatial_dims = len(kernel_shape)

        if stride is None:
            self.stride = self.kernel_shape
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

        num_spatial_dims = len(self.kernel_shape)
        batch_shape = x.shape[: -(num_spatial_dims + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial_dims + 1) :])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        x = lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, *self.kernel_shape, 1),
            window_strides=(1, *self.stride, 1),
            padding=padding,
        )

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial_dims + 1) :])

        return x


class AvgPool(Module):
    """N-dimensional average pooling.

    >>> pool = AvgPool(kernel_shape=(2, 2))
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    kernel_shape: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[tuple[int, int], ...]

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: str | int | tuple[int, ...] = 0,
    ) -> None:

        if isinstance(kernel_shape, int):
            raise TypeError(
                f"kernel_shape must be a tuple of ints (e.g. (2,) or (2, 2)), got int {kernel_shape}"
            )
        if len(kernel_shape) < 1:
            raise ValueError("kernel_shape must have at least one element")

        self.kernel_shape = kernel_shape

        num_spatial_dims = len(kernel_shape)

        if stride is None:
            self.stride = self.kernel_shape
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

        num_spatial_dims = len(self.kernel_shape)
        batch_shape = x.shape[: -(num_spatial_dims + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial_dims + 1) :])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        window_dims = (1, *self.kernel_shape, 1)
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

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial_dims + 1) :])

        return x
