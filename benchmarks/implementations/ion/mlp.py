"""Ion MLP benchmark."""

import ion.nn as nn
from ion.typing import PRNGKey

from ...configs import ModelConfig


def create_model(config: ModelConfig, *, key: PRNGKey) -> nn.MLP:
    """Create the benchmark model."""
    dims = (config.input_dim, *((config.width,) * (config.depth - 1)), config.num_classes)
    return nn.MLP(dims, key=key)
