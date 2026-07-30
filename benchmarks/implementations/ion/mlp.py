"""Ion MLP benchmark."""

from jaxtyping import PRNGKeyArray

import ion.nn as nn

from ...configs import ModelConfig


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> nn.MLP:
    """Create the benchmark model."""
    dims = (config.input_dim, *((config.width,) * (config.depth - 1)), config.num_classes)
    return nn.MLP(dims, key=key)
