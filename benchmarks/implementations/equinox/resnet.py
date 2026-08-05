"""Equinox ResNet benchmark."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ...configs import ModelConfig


class ResBlock(eqx.Module):
    """Two-convolution residual block."""

    conv_1: eqx.nn.Conv2d
    norm_1: eqx.nn.BatchNorm
    conv_2: eqx.nn.Conv2d
    norm_2: eqx.nn.BatchNorm
    projection: eqx.nn.Conv2d | None
    projection_norm: eqx.nn.BatchNorm | None

    def __init__(self, in_channels: int, channels: int, stride: int, *, key) -> None:
        key_1, key_2, key_projection = jax.random.split(key, 3)
        self.conv_1 = eqx.nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, use_bias=False, key=key_1
        )
        self.norm_1 = eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch")
        self.conv_2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, use_bias=False, key=key_2)
        self.norm_2 = eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch")
        self.projection = (
            eqx.nn.Conv2d(
                in_channels, channels, 1, stride=stride, use_bias=False, key=key_projection
            )
            if stride != 1 or in_channels != channels
            else None
        )
        self.projection_norm = (
            eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch")
            if self.projection is not None
            else None
        )

    def __call__(
        self, x: Float[Array, "c h w"], state: eqx.nn.State
    ) -> tuple[Float[Array, "c h w"], eqx.nn.State]:
        residual = x
        x, state = self.norm_1(self.conv_1(x), state)
        x, state = self.norm_2(self.conv_2(jax.nn.relu(x)), state)
        if self.projection is not None:
            assert self.projection_norm is not None
            residual, state = self.projection_norm(self.projection(residual), state)
        return jax.nn.relu(x + residual), state


class ResNet(eqx.Module):
    """Batch-normalized residual image classifier."""

    stem: eqx.nn.Conv2d
    stem_norm: eqx.nn.BatchNorm
    pool: eqx.nn.MaxPool2d
    blocks: tuple[ResBlock, ...]
    head: eqx.nn.Linear

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        keys = iter(jax.random.split(key, sum(config.block_depths) + 2))
        width = config.resnet_width
        self.stem = eqx.nn.Conv2d(3, width, 7, stride=2, padding=3, use_bias=False, key=next(keys))
        self.stem_norm = eqx.nn.BatchNorm(width, axis_name="batch", mode="batch")
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

    def __call__(
        self, x: Float[Array, "c h w"], state: eqx.nn.State
    ) -> tuple[Float[Array, " classes"], eqx.nn.State]:
        x, state = self.stem_norm(self.stem(x), state)
        x = self.pool(jax.nn.relu(x))
        for block in self.blocks:
            x, state = block(x, state)
        return self.head(jnp.mean(x, axis=(1, 2))), state


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> tuple[ResNet, eqx.nn.State]:
    """Create the benchmark model."""
    return eqx.nn.make_with_state(ResNet)(config, key=key)


def forward(model: ResNet, state: eqx.nn.State, inputs: Array) -> tuple[Array, eqx.nn.State]:
    """Apply the model to a batch, normalizing over the vmapped batch axis."""
    mapped = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    return mapped(inputs, state)
