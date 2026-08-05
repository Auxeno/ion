"""Ion ResNet benchmark."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import ion.nn as nn

from ...configs import ModelConfig


class ResBlock(nn.Module):
    """Two-convolution residual block."""

    conv_1: nn.Conv
    norm_1: nn.GroupNorm
    conv_2: nn.Conv
    norm_2: nn.GroupNorm
    projection: nn.Conv | None
    projection_norm: nn.GroupNorm | None

    def __init__(self, in_channels: int, channels: int, stride: int, *, key) -> None:
        key_1, key_2, key_projection = jax.random.split(key, 3)
        self.conv_1 = nn.Conv(
            in_channels, channels, (3, 3), stride=stride, padding=1, use_bias=False, key=key_1
        )
        self.norm_1 = nn.GroupNorm(channels, min(32, channels), 2)
        self.conv_2 = nn.Conv(channels, channels, (3, 3), padding=1, use_bias=False, key=key_2)
        self.norm_2 = nn.GroupNorm(channels, min(32, channels), 2)
        self.projection = (
            nn.Conv(
                in_channels, channels, (1, 1), stride=stride, use_bias=False, key=key_projection
            )
            if stride != 1 or in_channels != channels
            else None
        )
        self.projection_norm = (
            nn.GroupNorm(channels, min(32, channels), 2) if self.projection is not None else None
        )

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h w c"]:
        residual = x
        x = jax.nn.relu(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        if self.projection is not None:
            assert self.projection_norm is not None
            residual = self.projection_norm(self.projection(residual))
        return jax.nn.relu(x + residual)


class ResNet(nn.Module):
    """Group-normalized residual image classifier."""

    stem: nn.Conv
    stem_norm: nn.GroupNorm
    pool: nn.MaxPool
    blocks: tuple[ResBlock, ...]
    head: nn.Linear

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        num_blocks = sum(config.block_depths)
        keys = iter(jax.random.split(key, num_blocks + 2))
        width = config.resnet_width
        self.stem = nn.Conv(3, width, (7, 7), stride=2, padding=3, use_bias=False, key=next(keys))
        self.stem_norm = nn.GroupNorm(width, min(32, width), 2)
        self.pool = nn.MaxPool((3, 3), stride=2, padding=1)

        in_channels = width
        blocks = []
        channels_by_stage = tuple(width * 2**stage for stage in range(4))
        for stage, (channels, depth) in enumerate(zip(channels_by_stage, config.block_depths)):
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(ResBlock(in_channels, channels, stride, key=next(keys)))
                in_channels = channels
        self.blocks = tuple(blocks)
        self.head = nn.Linear(in_channels, config.num_classes, key=next(keys))

    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b classes"]:
        x = self.pool(jax.nn.relu(self.stem_norm(self.stem(x))))
        for block in self.blocks:
            x = block(x)
        return self.head(jnp.mean(x, axis=(1, 2)))


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> ResNet:
    """Create the benchmark model."""
    return ResNet(config, key=key)
