"""Pooling layers.

Modules:
    MaxPool  Downsample by taking the max over windows.
    AvgPool  Downsample by averaging over windows.

Channels-last format: (..., spatial, channels).
Stride defaults to kernel shape when not specified.
"""

import math
from typing import Literal

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..module import Module


class MaxPool(Module):
    """N-dimensional max pooling.

    >>> pool = MaxPool(kernel_shape=(2, 2))
    >>> pool(x)  # (b, h, w, c) -> (b, h // 2, w // 2, c)
    """

    kernel_shape: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Literal["SAME", "VALID"] | tuple[tuple[int, int], ...]

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        *,
        stride: int | tuple[int, ...] | None = None,
        padding: Literal["SAME", "VALID"] | int | tuple[int, ...] = 0,
    ) -> None:

        if len(kernel_shape) < 1:
            raise ValueError("kernel_shape must have at least one element")

        num_spatial = len(kernel_shape)

        if stride is None:
            stride = kernel_shape
        elif isinstance(stride, int):
            stride = (stride,) * num_spatial

        if isinstance(padding, str):
            resolved_padding = padding
        elif isinstance(padding, int):
            resolved_padding = tuple((padding, padding) for _ in range(num_spatial))
        else:
            resolved_padding = tuple((p, p) for p in padding)

        # A window landing entirely in padding has no real elements to reduce over
        if not isinstance(resolved_padding, str) and any(
            lo >= k or hi >= k for (lo, hi), k in zip(resolved_padding, kernel_shape)
        ):
            raise ValueError(
                f"padding ({resolved_padding}) must be smaller than kernel_shape ({kernel_shape})"
            )

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = resolved_padding

    def __call__(self, x: Float[Array, "b ... c"]) -> Float[Array, "b ... c"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        x = lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=lax.max,
            window_dimensions=(1, *self.kernel_shape, 1),
            window_strides=(1, *self.stride, 1),
            padding=padding,
        )

        return x.astype(dtype)


class AvgPool(Module):
    """N-dimensional average pooling.

    >>> pool = AvgPool(kernel_shape=(2, 2))
    >>> pool(x)  # (b, h, w, c) -> (b, h // 2, w // 2, c)
    """

    kernel_shape: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Literal["SAME", "VALID"] | tuple[tuple[int, int], ...]
    count_include_pad: bool

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        *,
        stride: int | tuple[int, ...] | None = None,
        padding: Literal["SAME", "VALID"] | int | tuple[int, ...] = 0,
        count_include_pad: bool = True,
    ) -> None:

        if len(kernel_shape) < 1:
            raise ValueError("kernel_shape must have at least one element")

        num_spatial = len(kernel_shape)

        if stride is None:
            stride = kernel_shape
        elif isinstance(stride, int):
            stride = (stride,) * num_spatial

        if isinstance(padding, str):
            resolved_padding = padding
        elif isinstance(padding, int):
            resolved_padding = tuple((padding, padding) for _ in range(num_spatial))
        else:
            resolved_padding = tuple((p, p) for p in padding)

        # A window landing entirely in padding has no real elements to reduce over
        if not isinstance(resolved_padding, str) and any(
            lo >= k or hi >= k for (lo, hi), k in zip(resolved_padding, kernel_shape)
        ):
            raise ValueError(
                f"padding ({resolved_padding}) must be smaller than kernel_shape ({kernel_shape})"
            )

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = resolved_padding
        self.count_include_pad = count_include_pad

    def __call__(self, x: Float[Array, "b ... c"]) -> Float[Array, "b ... c"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)

        padding = self.padding if isinstance(self.padding, str) else ((0, 0), *self.padding, (0, 0))

        window_dims = (1, *self.kernel_shape, 1)
        window_strides = (1, *self.stride, 1)

        # Counted before the sum, since windows overlapping padding hold fewer real elements
        if self.count_include_pad:
            window_counts = math.prod(self.kernel_shape)
        else:
            window_counts = lax.reduce_window(
                operand=jnp.ones_like(x),
                init_value=0.0,
                computation=lax.add,
                window_dimensions=window_dims,
                window_strides=window_strides,
                padding=padding,
            )

        x = lax.reduce_window(
            operand=x,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=window_dims,
            window_strides=window_strides,
            padding=padding,
        )

        x = x / window_counts

        return x.astype(dtype)
