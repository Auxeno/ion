"""Upsampling layers.

Modules:
    Upsample  N-dimensional upsampling via interpolation.

Channels-last format: (..., spatial, channels).
Nearest-neighbor interpolation by default.
"""

import jax
from jaxtyping import Array, Float

from ..module import Module


class Upsample(Module):
    """N-dimensional upsampling via interpolation.

    >>> up = Upsample(2, scale_factor=2)
    >>> up(x)  # (*, h, w, c) -> (*, h * 2, w * 2, c)
    """

    num_spatial_dims: int
    scale_factor: tuple[int, ...]
    mode: str

    def __init__(
        self,
        num_spatial_dims: int,
        scale_factor: int | tuple[int, ...],
        mode: str = "nearest",
    ) -> None:

        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims ({num_spatial_dims}) must be >= 1")

        self.num_spatial_dims = num_spatial_dims

        if isinstance(scale_factor, int):
            self.scale_factor = (scale_factor,) * num_spatial_dims
        else:
            self.scale_factor = scale_factor

        self.mode = mode

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        num_spatial = self.num_spatial_dims
        batch_shape = x.shape[:-(num_spatial + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial + 1):])

        spatial = x.shape[1:-1]
        new_spatial = tuple(s * f for s, f in zip(spatial, self.scale_factor))
        new_shape = (x.shape[0], *new_spatial, x.shape[-1])
        x = jax.image.resize(x, new_shape, method=self.mode)

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial + 1):])

        return x
