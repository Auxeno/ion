"""Benchmark model configurations."""

from dataclasses import dataclass
from typing import Literal

ModelName = Literal["mlp", "resnet", "gpt"]
ModelSize = Literal["tiny", "small", "medium"]

MODELS: tuple[ModelName, ...] = ("mlp", "resnet", "gpt")
SIZES: tuple[ModelSize, ...] = ("tiny", "small", "medium")
MODEL_LABELS: dict[ModelName, str] = {
    "mlp": "MLP",
    "resnet": "ResNet",
    "gpt": "GPT",
}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by every framework implementation."""

    model: ModelName
    size: ModelSize
    batch_size: int
    num_classes: int = 1000
    input_dim: int = 1024
    width: int = 0
    depth: int = 0
    image_size: int = 224
    block_depths: tuple[int, ...] = ()
    resnet_width: int = 64
    vocab_size: int = 32_000
    seq_len: int = 512
    num_heads: int = 0

    @property
    def units_per_step(self) -> int:
        """Samples, or tokens for GPT, processed by one step."""
        if self.model == "gpt":
            return self.batch_size * self.seq_len
        return self.batch_size


_MLP: dict[ModelSize, ModelConfig] = {
    "tiny": ModelConfig("mlp", "tiny", batch_size=2048, width=128, depth=4),
    "small": ModelConfig("mlp", "small", batch_size=1024, width=512, depth=8),
    "medium": ModelConfig("mlp", "medium", batch_size=512, width=2048, depth=12),
}

_RESNET: dict[ModelSize, ModelConfig] = {
    "tiny": ModelConfig(
        "resnet",
        "tiny",
        batch_size=128,
        block_depths=(1, 1, 1, 1),
        resnet_width=32,
    ),
    "small": ModelConfig(
        "resnet",
        "small",
        batch_size=64,
        block_depths=(2, 2, 2, 2),
    ),
    "medium": ModelConfig(
        "resnet",
        "medium",
        batch_size=32,
        block_depths=(3, 4, 23, 3),
    ),
}

_GPT: dict[ModelSize, ModelConfig] = {
    "tiny": ModelConfig(
        "gpt",
        "tiny",
        batch_size=32,
        width=128,
        depth=2,
        seq_len=128,
        num_heads=4,
    ),
    "small": ModelConfig(
        "gpt",
        "small",
        batch_size=16,
        width=384,
        depth=6,
        seq_len=256,
        num_heads=6,
    ),
    "medium": ModelConfig("gpt", "medium", batch_size=8, width=768, depth=12, num_heads=12),
}

CONFIGS: dict[ModelName, dict[ModelSize, ModelConfig]] = {
    "mlp": _MLP,
    "resnet": _RESNET,
    "gpt": _GPT,
}


def get_config(model: ModelName, size: ModelSize) -> ModelConfig:
    """Return a benchmark model configuration."""
    return CONFIGS[model][size]
