"""Flax NNX ResNet benchmark."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from ...configs import ModelConfig


class ResBlock(nnx.Module):
    """Two-convolution residual block."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv_1 = nnx.Conv(
            in_channels,
            channels,
            (3, 3),
            strides=stride,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.norm_1 = nnx.GroupNorm(
            channels,
            num_groups=min(32, channels),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.conv_2 = nnx.Conv(
            channels,
            channels,
            (3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.norm_2 = nnx.GroupNorm(
            channels,
            num_groups=min(32, channels),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.projection = (
            nnx.Conv(
                in_channels,
                channels,
                (1, 1),
                strides=stride,
                use_bias=False,
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
            if stride != 1 or in_channels != channels
            else None
        )
        self.projection_norm = (
            nnx.GroupNorm(
                channels,
                num_groups=min(32, channels),
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
            if self.projection is not None
            else None
        )

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h w c"]:
        residual = x
        x = jax.nn.relu(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        if self.projection is not None:
            assert self.projection_norm is not None
            residual = self.projection_norm(self.projection(residual))
        return jax.nn.relu(x + residual)


class ResNet(nnx.Module):
    """Group-normalized residual image classifier."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        width = config.resnet_width
        self.stem = nnx.Conv(
            3,
            width,
            (7, 7),
            strides=2,
            padding=((3, 3), (3, 3)),
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.stem_norm = nnx.GroupNorm(
            width, num_groups=min(32, width), dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs
        )
        in_channels = width
        blocks = []
        channels_by_stage = tuple(width * 2**stage for stage in range(4))
        for stage, (channels, depth) in enumerate(zip(channels_by_stage, config.block_depths)):
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(ResBlock(in_channels, channels, stride, rngs=rngs))
                in_channels = channels
        self.blocks = nnx.List(blocks)
        self.head = nnx.Linear(
            in_channels, config.num_classes, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b classes"]:
        x = nnx.max_pool(
            jax.nn.relu(self.stem_norm(self.stem(x))),
            (3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # pyright: ignore[reportArgumentType]
        )
        for block in self.blocks:
            x = block(x)
        return self.head(jnp.mean(x, axis=(1, 2)))


def create_model(config: ModelConfig, *, seed: int) -> ResNet:
    """Create the benchmark model."""
    return ResNet(config, rngs=nnx.Rngs(seed))


def forward(model: ResNet, inputs: Array) -> Array:
    """Apply the model to a batch."""
    return model(inputs)
