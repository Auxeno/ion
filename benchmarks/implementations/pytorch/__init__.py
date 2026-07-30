"""PyTorch benchmark adapter."""

from typing import Any

import torch
import torch.nn.functional as functional

from ...configs import ModelConfig
from ...protocol import Metric


class PyTorchWorkload:
    """PyTorch implementation of the benchmark protocol."""

    framework_version = torch.__version__
    software = {
        "torch": framework_version,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }

    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        torch.manual_seed(seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = __import__(
            f"benchmarks.implementations.pytorch.{config.model}",
            fromlist=["create_model"],
        )
        self.model = module.create_model(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
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
        self.batch = (inputs, targets)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def _forward(self, model, batch):
        inputs, _ = batch
        with self._autocast():
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

    def prepare(self, metric: Metric, *, compiled: bool):
        if metric == "forward":
            function = self._forward
            if compiled:
                function = torch.compile(function)
            return lambda: function(self.model, self.batch)
        if metric == "forward_backward":
            function = self._forward_backward
            if compiled:
                function = torch.compile(function)
            return lambda: function(self.model, self.batch)

        function = self._full_step
        if compiled:
            function = torch.compile(function)

        def operation():
            return function(self.model, self.optimizer, self.batch)

        return operation

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


def create_workload(config: ModelConfig, *, seed: int) -> PyTorchWorkload:
    """Create a PyTorch benchmark workload."""
    return PyTorchWorkload(config, seed=seed)


def device_name() -> str:
    """Return the active PyTorch device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return str(torch.device("cpu"))
