"""PyTorch benchmark adapter."""

import importlib
from typing import Any

import torch
import torch.nn.functional as functional

from ...configs import ModelConfig
from ...protocol import Operation


class Workload:
    """PyTorch benchmark workload."""

    framework_version = torch.__version__
    software = {
        "torch": framework_version,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }

    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        # Build model and optimizer state
        torch.manual_seed(seed)
        module = importlib.import_module(f"{__package__}.{config.model}")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else str(self.device)
        )
        self.model = module.create_model(config).to(self.device).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

        # Generate fixed inputs and targets outside the timed region
        if config.model == "gpt":
            inputs = torch.randint(
                config.vocab_size,
                (config.batch_size, config.seq_len),
                device=self.device,
            )
            targets = torch.randint(
                config.vocab_size,
                (config.batch_size, config.seq_len),
                device=self.device,
            )
        else:
            shape = (
                (config.batch_size, 3, config.image_size, config.image_size)
                if config.model == "resnet"
                else (config.batch_size, config.input_dim)
            )
            inputs = torch.randn(shape, dtype=torch.bfloat16, device=self.device)
            targets = torch.randint(config.num_classes, (config.batch_size,), device=self.device)
        self.batch = inputs, targets

    def _forward(self, model, batch):
        inputs, _ = batch
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return model(inputs)

    def _loss(self, model, batch):
        _, targets = batch
        logits = self._forward(model, batch).float()
        if self.config.model == "gpt":
            return functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return functional.cross_entropy(logits, targets)

    def _forward_backward(self, model, batch):
        model.zero_grad(set_to_none=True)
        loss = self._loss(model, batch)
        loss.backward()
        return loss

    def _full_step(self, model, optimizer, batch):
        optimizer.zero_grad(set_to_none=True)
        loss = self._loss(model, batch)
        loss.backward()
        optimizer.step()
        return loss

    def prepare(self, operation: Operation, *, compiled: bool):
        # Select and optionally compile the requested operation
        function = getattr(self, f"_{operation}")
        if compiled:
            function = torch.compile(function)

        if operation == "full_step":
            return lambda: function(self.model, self.optimizer, self.batch)
        return lambda: function(self.model, self.batch)

    def synchronize(self, value: Any) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def peak_memory(self) -> int | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device)

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
