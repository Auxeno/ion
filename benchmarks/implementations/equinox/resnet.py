"""Equinox ResNet benchmark."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ...configs import ModelConfig


class ResBlock(eqx.Module):
    """Two-convolution residual block."""

    conv_1: eqx.nn.Conv2d
    norm_1: eqx.nn.GroupNorm
    conv_2: eqx.nn.Conv2d
    norm_2: eqx.nn.GroupNorm
    projection: eqx.nn.Conv2d | None
    projection_norm: eqx.nn.GroupNorm | None

    def __init__(self, in_channels: int, channels: int, stride: int, *, key) -> None:
        key_1, key_2, key_projection = jax.random.split(key, 3)
        self.conv_1 = eqx.nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, use_bias=False, key=key_1
        )
        self.norm_1 = eqx.nn.GroupNorm(min(32, channels), channels)
        self.conv_2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, use_bias=False, key=key_2)
        self.norm_2 = eqx.nn.GroupNorm(min(32, channels), channels)
        self.projection = (
            eqx.nn.Conv2d(
                in_channels, channels, 1, stride=stride, use_bias=False, key=key_projection
            )
            if stride != 1 or in_channels != channels
            else None
        )
        self.projection_norm = (
            eqx.nn.GroupNorm(min(32, channels), channels) if self.projection is not None else None
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        residual = x
        x = jax.nn.relu(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        if self.projection is not None:
            assert self.projection_norm is not None
            residual = self.projection_norm(self.projection(residual))
        return jax.nn.relu(x + residual)


class ResNet(eqx.Module):
    """Group-normalized residual image classifier."""

    stem: eqx.nn.Conv2d
    stem_norm: eqx.nn.GroupNorm
    pool: eqx.nn.MaxPool2d
    blocks: tuple[ResBlock, ...]
    head: eqx.nn.Linear

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        keys = iter(jax.random.split(key, sum(config.block_depths) + 2))
        width = config.resnet_width
        self.stem = eqx.nn.Conv2d(3, width, 7, stride=2, padding=3, use_bias=False, key=next(keys))
        self.stem_norm = eqx.nn.GroupNorm(min(32, width), width)
        self.pool = eqx.nn.MaxPool2d(3, stride=2, padding=1)

        in_channels = width
        blocks = []
        channels_by_stage = tuple(width * 2**stage for stage in range(4))
        for stage, (channels, depth) in enumerate(zip(channels_by_stage, config.block_depths)):
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(ResBlock(in_channels, channels, stride, key=next(keys)))
                in_channels = channels
        self.blocks = tuple(blocks)
        self.head = eqx.nn.Linear(in_channels, config.num_classes, key=next(keys))

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, " classes"]:
        x = self.pool(jax.nn.relu(self.stem_norm(self.stem(x))))
        for block in self.blocks:
            x = block(x)
        return self.head(jnp.mean(x, axis=(1, 2)))


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> ResNet:
    """Create the benchmark model."""
    return ResNet(config, key=key)


def forward(model: ResNet, inputs: Array) -> Array:
    """Apply the model to a batch."""
    return jax.vmap(model)(inputs)
