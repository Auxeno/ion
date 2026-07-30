"""PyTorch MLP benchmark."""

from torch import nn

from ...configs import ModelConfig


def create_model(config: ModelConfig) -> nn.Sequential:
    """Create the benchmark model."""
    dims = (config.input_dim, *((config.width,) * (config.depth - 1)), config.num_classes)
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < config.depth - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
