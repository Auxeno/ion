"""Pooling layers.

Modules:
    MaxPool1d  Downsample sequences by taking the max over windows.
    MaxPool2d  Downsample images by taking the max over windows.
    AvgPool1d  Downsample sequences by averaging over windows.
    AvgPool2d  Downsample images by averaging over windows.

Channels-last format: (..., spatial, channels).
Stride defaults to kernel size when not specified.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..module import Module


class MaxPool1d(Module):
    """1D max pooling.

    >>> pool = MaxPool1d(kernel_size=2)
    >>> pool(x)  # (*, length, c) -> (*, length // 2, c)
    """

    kernel_size: int
    stride: int
    padding: str | tuple[tuple[int, int]]

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: str | int = 0,
    ) -> None:

        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        if isinstance(padding, int):
            self.padding = ((padding, padding),)
        else:
            self.padding = padding

    def __call__(self, x: Float[Array, "... l c"]) -> Float[Array, "... l c"]:

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        x = lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, self.kernel_size, 1),
            window_strides=(1, self.stride, 1),
            padding=padding,
        )

        x = x.reshape(*batch_shape, *x.shape[-2:])

        return x


class AvgPool1d(Module):
    """1D average pooling.

    >>> pool = AvgPool1d(kernel_size=2)
    >>> pool(x)  # (*, length, c) -> (*, length // 2, c)
    """

    kernel_size: int
    stride: int
    padding: str | tuple[tuple[int, int]]

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: str | int = 0,
    ) -> None:

        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        if isinstance(padding, int):
            self.padding = ((padding, padding),)
        else:
            self.padding = padding

    def __call__(self, x: Float[Array, "... l c"]) -> Float[Array, "... l c"]:

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        window_dims = (1, self.kernel_size, 1)
        window_strides = (1, self.stride, 1)

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

        x = x.reshape(*batch_shape, *x.shape[-2:])

        return x


class MaxPool2d(Module):
    """2D max pooling.

    >>> pool = MaxPool2d(kernel_size=2)
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[tuple[int, int], tuple[int, int]]

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: str | int | tuple[int, int] = 0,
    ) -> None:

        if isinstance(kernel_size, int):
            kh, kw = (kernel_size, kernel_size)
        else:
            kh, kw = kernel_size
        self.kernel_size = (kh, kw)

        if stride is None:
            self.stride = (kh, kw)
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif isinstance(padding, tuple):
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            self.padding = padding

    def __call__(self, x: Float[Array, "... h w c"]) -> Float[Array, "... h w c"]:

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        # Max reduction
        x = lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, *self.kernel_size, 1),
            window_strides=(1, *self.stride, 1),
            padding=padding,
        )

        x = x.reshape(*batch_shape, *x.shape[-3:])

        return x


class AvgPool2d(Module):
    """2D average pooling.

    >>> pool = AvgPool2d(kernel_size=2)
    >>> pool(x)  # (*, h, w, c) -> (*, h // 2, w // 2, c)
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: str | tuple[tuple[int, int], tuple[int, int]]

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: str | int | tuple[int, int] = 0,
    ) -> None:

        if isinstance(kernel_size, int):
            kh, kw = (kernel_size, kernel_size)
        else:
            kh, kw = kernel_size
        self.kernel_size = (kh, kw)

        if stride is None:
            self.stride = (kh, kw)
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif isinstance(padding, tuple):
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            self.padding = padding

    def __call__(self, x: Float[Array, "... h w c"]) -> Float[Array, "... h w c"]:

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

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

        x = x.reshape(*batch_shape, *x.shape[-3:])

        return x
