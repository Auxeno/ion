"""PyTorch ResNet benchmark."""

import torch
from torch import Tensor, nn

from ...configs import ModelConfig


class ResBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, in_channels: int, channels: int, stride: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.norm_1 = nn.GroupNorm(min(32, channels), channels)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm_2 = nn.GroupNorm(min(32, channels), channels)
        self.projection = (
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != channels
            else None
        )
        self.projection_norm = (
            nn.GroupNorm(min(32, channels), channels) if self.projection is not None else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = torch.relu(self.norm_1(self.conv_1(x)))
        x = self.norm_2(self.conv_2(x))
        if self.projection is not None:
            assert self.projection_norm is not None
            residual = self.projection_norm(self.projection(residual))
        return torch.relu(x + residual)


class ResNet(nn.Module):
    """Group-normalized residual image classifier."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.resnet_width
        self.stem = nn.Conv2d(3, width, 7, stride=2, padding=3, bias=False)
        self.stem_norm = nn.GroupNorm(min(32, width), width)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        in_channels = width
        blocks = []
        channels_by_stage = tuple(width * 2**stage for stage in range(4))
        for stage, (channels, depth) in enumerate(zip(channels_by_stage, config.block_depths)):
            for index in range(depth):
                stride = 2 if stage > 0 and index == 0 else 1
                blocks.append(ResBlock(in_channels, channels, stride))
                in_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(in_channels, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(torch.relu(self.stem_norm(self.stem(x))))
        for block in self.blocks:
            x = block(x)
        return self.head(torch.mean(x, dim=(2, 3)))


def create_model(config: ModelConfig) -> ResNet:
    """Create the benchmark model."""
    return ResNet(config)
