"""Equinox MLP benchmark."""

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from ...configs import ModelConfig


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> eqx.nn.MLP:
    """Create the benchmark model."""
    return eqx.nn.MLP(
        config.input_dim,
        config.num_classes,
        config.width,
        config.depth - 1,
        activation=jax.nn.relu,
        key=key,
    )


def forward(model: eqx.nn.MLP, inputs: Array) -> Array:
    """Apply the model to a batch."""
    return jax.vmap(model)(inputs)
