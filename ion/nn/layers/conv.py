"""Convolutional layers.

Modules:
    Conv           N-dimensional convolution.
    ConvTranspose  N-dimensional transposed convolution.

Channels-last format to match image data conventions: (..., spatial, channels).
He normal weight init for ReLU activation, zeros for bias.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class Conv(Module):
    """N-dimensional convolution layer.

    >>> Conv(1, 3, 16, kernel_size=5, padding="SAME", key=key)  # Conv1d
    >>> Conv(2, 3, 16, kernel_size=3, padding=1, key=key)       # Conv2d
    >>> conv(x)  # (*, h, w, 3) -> (*, h, w, 16)
    """

    w: Param[Float[Array, "..."]]
    b: Param[Float[Array, " c"]] | None
    num_spatial_dims: int
    stride: tuple[int, ...]
    padding: str | tuple[tuple[int, int], ...]
    dilation: tuple[int, ...]
    groups: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims ({num_spatial_dims}) must be >= 1")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        kernel_size = (
            (kernel_size,) * num_spatial_dims if isinstance(kernel_size, int) else kernel_size
        )

        self.num_spatial_dims = num_spatial_dims
        self.stride = (stride,) * num_spatial_dims if isinstance(stride, int) else stride
        self.dilation = (dilation,) * num_spatial_dims if isinstance(dilation, int) else dilation
        self.groups = groups

        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        else:
            self.padding = tuple((p, p) for p in padding)

        key_w, key_b = jax.random.split(key)
        self.w = Param(
            w_init(
                shape=(*kernel_size, in_channels // groups, out_channels), dtype=dtype, key=key_w
            )
        )
        self.b = Param(b_init(shape=(out_channels,), dtype=dtype, key=key_b)) if bias else None

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        num_spatial = self.num_spatial_dims
        batch_shape = x.shape[: -(num_spatial + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial + 1) :])

        spatial_dims = tuple(range(1, num_spatial + 1))
        lhs_spec = (0, num_spatial + 1) + spatial_dims
        rhs_spec = (num_spatial + 1, num_spatial) + tuple(range(num_spatial))

        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.w.value,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            dimension_numbers=lax.ConvDimensionNumbers(lhs_spec, rhs_spec, lhs_spec),
            feature_group_count=self.groups,
        )

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial + 1) :])

        if self.b is not None:
            x = x + self.b

        return x


class ConvTranspose(Module):
    """N-dimensional transposed convolution layer.

    >>> ConvTranspose(1, 3, 16, kernel_size=5, padding=2, key=key)        # ConvTranspose1d
    >>> ConvTranspose(2, 3, 16, kernel_size=3, padding="VALID", key=key)  # ConvTranspose2d
    >>> conv_t(x)  # (*, h, w, 3) -> (*, h, w, 16)
    """

    w: Param[Float[Array, "..."]]
    b: Param[Float[Array, " c"]] | None
    num_spatial_dims: int
    stride: tuple[int, ...]
    padding: tuple[tuple[int, int], ...]
    dilation: tuple[int, ...]
    groups: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims ({num_spatial_dims}) must be >= 1")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        kernel_size = (
            (kernel_size,) * num_spatial_dims if isinstance(kernel_size, int) else kernel_size
        )
        stride = (stride,) * num_spatial_dims if isinstance(stride, int) else stride
        output_padding = (
            (output_padding,) * num_spatial_dims
            if isinstance(output_padding, int)
            else output_padding
        )
        dilation = (dilation,) * num_spatial_dims if isinstance(dilation, int) else dilation

        for s, o in zip(stride, output_padding):
            if o >= s:
                raise ValueError(
                    f"output_padding ({output_padding}) must be less than "
                    f"stride ({stride}) elementwise"
                )

        self.num_spatial_dims = num_spatial_dims
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Compute transposed padding
        if isinstance(padding, str):
            if padding == "VALID":
                p0 = (0,) * num_spatial_dims
                p1 = (0,) * num_spatial_dims
            else:  # SAME
                p_sums = tuple(
                    d * (k - 1) - s + a + 1
                    for k, s, a, d in zip(kernel_size, stride, output_padding, dilation)
                )
                p0 = tuple(ps // 2 for ps in p_sums)
                p1 = tuple(ps - p0i for ps, p0i in zip(p_sums, p0))
        elif isinstance(padding, int):
            p0 = p1 = (padding,) * num_spatial_dims
        else:
            p0 = p1 = tuple(padding)

        dk = tuple(d * (k - 1) for k, d in zip(kernel_size, dilation))
        self.padding = tuple(
            (dk_i - a, dk_i - b + op) for dk_i, a, b, op in zip(dk, p0, p1, output_padding)
        )

        key_w, key_b = jax.random.split(key)
        self.w = Param(
            w_init(
                shape=(*kernel_size, in_channels // groups, out_channels), dtype=dtype, key=key_w
            )
        )
        self.b = Param(b_init(shape=(out_channels,), dtype=dtype, key=key_b)) if bias else None

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        num_spatial = self.num_spatial_dims
        batch_shape = x.shape[: -(num_spatial + 1)]
        x = x.reshape(-1, *x.shape[-(num_spatial + 1) :])

        spatial_dims = tuple(range(1, num_spatial + 1))
        lhs_spec = (0, num_spatial + 1) + spatial_dims
        rhs_spec = (num_spatial + 1, num_spatial) + tuple(range(num_spatial))

        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.w.value,
            window_strides=(1,) * num_spatial,
            padding=self.padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            dimension_numbers=lax.ConvDimensionNumbers(lhs_spec, rhs_spec, lhs_spec),
            feature_group_count=self.groups,
        )

        x = x.reshape(*batch_shape, *x.shape[-(num_spatial + 1) :])

        if self.b is not None:
            x = x + self.b

        return x
