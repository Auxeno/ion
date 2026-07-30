"""Flax NNX MLP benchmark."""

import jax
from flax import nnx
from jaxtyping import Array

from ...configs import ModelConfig


def create_model(config: ModelConfig, *, seed: int) -> nnx.Sequential:
    """Create the benchmark model."""
    dims = (
        config.input_dim,
        *((config.width,) * (config.depth - 1)),
        config.num_classes,
    )
    rngs = nnx.Rngs(seed)
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(
            nnx.Linear(
                in_dim, out_dim, dtype=jax.numpy.bfloat16, param_dtype=jax.numpy.float32, rngs=rngs
            )
        )
        if index < config.depth - 1:
            layers.append(jax.nn.relu)
    return nnx.Sequential(*layers)


def forward(model: nnx.Sequential, inputs: Array) -> Array:
    """Apply the model to a batch."""
    return model(inputs)
