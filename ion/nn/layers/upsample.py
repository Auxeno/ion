"""Upsampling layers.

Modules:
    Upsample1d  1D upsampling via interpolation.
    Upsample2d  2D upsampling via interpolation.

Channels-last format: (..., spatial, channels).
Nearest-neighbor interpolation by default.
"""

import jax
from jaxtyping import Array, Float

from ..module import Module


class Upsample1d(Module):
    """1D upsampling via interpolation.

    >>> up = Upsample1d(scale_factor=2)
    >>> up(x)  # (*, t, c) -> (*, t * 2, c)
    """

    scale_factor: int
    mode: str

    def __init__(self, scale_factor: int, mode: str = "nearest") -> None:

        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: Float[Array, "... l c"]) -> Float[Array, "... l c"]:

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])

        b, t, c = x.shape
        new_shape = (b, t * self.scale_factor, c)
        x = jax.image.resize(x, new_shape, method=self.mode)

        x = x.reshape(*batch_shape, *x.shape[-2:])

        return x


class Upsample2d(Module):
    """2D upsampling via interpolation.

    >>> up = Upsample2d(scale_factor=2)
    >>> up(x)  # (*, h, w, c) -> (*, h * 2, w * 2, c)
    """

    scale_factor: tuple[int, int]
    mode: str

    def __init__(self, scale_factor: int | tuple[int, int], mode: str = "nearest") -> None:

        if isinstance(scale_factor, int):
            self.scale_factor = (scale_factor, scale_factor)
        else:
            self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: Float[Array, "... h w c"]) -> Float[Array, "... h w c"]:

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        b, h, w, c = x.shape
        sh, sw = self.scale_factor
        new_shape = (b, h * sh, w * sw, c)
        x = jax.image.resize(x, new_shape, method=self.mode)

        x = x.reshape(*batch_shape, *x.shape[-3:])

        return x
