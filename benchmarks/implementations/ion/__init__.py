"""Ion benchmark adapter."""

import importlib
from importlib import metadata
from typing import Any

import jax
import jax.numpy as jnp
import optax

import ion

from ...configs import ModelConfig
from ...protocol import Operation

try:
    ION_VERSION = metadata.version("ion-nn")
except metadata.PackageNotFoundError:
    ION_VERSION = "local"


class Workload:
    """Ion benchmark workload."""

    framework_version = ION_VERSION
    software = {
        "ion": framework_version,
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
    }

    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        # Build model and optimizer state
        module = importlib.import_module(f"{__package__}.{config.model}")
        key_model, key_inputs, key_targets = jax.random.split(jax.random.key(seed), 3)

        self.config = config
        self.model = module.create_model(config, key=key_model)
        self.optimizer = ion.Optimizer(optax.adamw(3e-4), self.model)
        self.parameter_count = self.model.num_params

        device = jax.devices()[0]
        self.device_name = getattr(device, "device_kind", str(device))

        # Generate fixed inputs and targets outside the timed region
        if config.model == "gpt":
            inputs = jax.random.randint(
                key_inputs,
                (config.batch_size, config.seq_len),
                0,
                config.vocab_size,
            )
            targets = jax.random.randint(
                key_targets,
                (config.batch_size, config.seq_len),
                0,
                config.vocab_size,
            )
        else:
            shape = (
                (config.batch_size, config.image_size, config.image_size, 3)
                if config.model == "resnet"
                else (config.batch_size, config.input_dim)
            )
            inputs = jax.random.normal(key_inputs, shape, dtype=jnp.bfloat16)
            targets = jax.random.randint(key_targets, (config.batch_size,), 0, config.num_classes)
        self.batch = inputs, targets

    @staticmethod
    def _forward(model, batch):
        inputs, _ = batch
        return model.astype(jnp.bfloat16)(inputs)

    @classmethod
    def _loss(cls, model, batch):
        _, targets = batch
        logits = cls._forward(model, batch).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    @classmethod
    def _forward_backward(cls, model, batch):
        return jax.value_and_grad(cls._loss)(model, batch)

    @classmethod
    def _full_step(cls, model, optimizer, batch):
        loss, grads = jax.value_and_grad(cls._loss)(model, batch)
        model, optimizer = optimizer.update(model, grads)
        return model, optimizer, loss

    def prepare(self, operation: Operation, *, compiled: bool):
        # Select and optionally compile the requested operation
        function = getattr(self, f"_{operation}")
        if compiled:
            function = jax.jit(function)

        if operation != "full_step":
            return lambda: function(self.model, self.batch)

        def step():
            self.model, self.optimizer, loss = function(self.model, self.optimizer, self.batch)
            return loss

        return step

    def synchronize(self, value: Any) -> None:
        jax.block_until_ready(value)

    def peak_memory(self) -> int | None:
        stats = jax.devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use") if stats else None

    def reset_peak_memory(self) -> None:
        """JAX does not expose an allocator peak reset."""
