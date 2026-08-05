"""Equinox MLP benchmark."""

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from ...configs import ModelConfig


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> tuple[eqx.nn.MLP, None]:
    """Create the benchmark model."""
    model = eqx.nn.MLP(
        config.input_dim,
        config.num_classes,
        config.width,
        config.depth - 1,
        activation=jax.nn.relu,
        key=key,
    )
    return model, None


def forward(model: eqx.nn.MLP, state: None, inputs: Array) -> tuple[Array, None]:
    """Apply the model to a batch."""
    return jax.vmap(model)(inputs), state
