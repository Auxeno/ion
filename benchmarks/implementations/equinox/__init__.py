"""Equinox benchmark adapter."""

import importlib
from importlib import metadata
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ...configs import ModelConfig
from ...protocol import Operation


class Workload:
    """Equinox benchmark workload."""

    framework_version = eqx.__version__
    software = {
        "equinox": framework_version,
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
    }

    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        # Build model and optimizer state
        module = importlib.import_module(f"{__package__}.{config.model}")
        key_model, key_inputs, key_targets = jax.random.split(jax.random.key(seed), 3)

        self.config = config
        self.module = module
        self.model, self.state = module.create_model(config, key=key_model)
        self.transform = optax.adamw(3e-4)
        parameters = eqx.filter(self.model, eqx.is_inexact_array)
        self.optimizer = self.transform.init(parameters)
        self.parameter_count = sum(
            value.size for value in jax.tree.leaves(parameters) if value is not None
        )

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
                (config.batch_size, 3, config.image_size, config.image_size)
                if config.model == "resnet"
                else (config.batch_size, config.input_dim)
            )
            inputs = jax.random.normal(key_inputs, shape, dtype=jnp.bfloat16)
            targets = jax.random.randint(key_targets, (config.batch_size,), 0, config.num_classes)
        self.batch = inputs, targets

    @staticmethod
    def _cast(model):
        return jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            model,
        )

    def _forward(self, model, state, batch):
        inputs, _ = batch
        return self.module.forward(self._cast(model), state, inputs)

    def _loss(self, model, state, batch):
        _, targets = batch
        logits, state = self._forward(model, state, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), targets
        ).mean()
        return loss, state

    def _forward_backward(self, model, state, batch):
        return eqx.filter_value_and_grad(self._loss, has_aux=True)(model, state, batch)

    def _full_step(self, model, state, optimizer, batch):
        (loss, state), grads = eqx.filter_value_and_grad(self._loss, has_aux=True)(
            model, state, batch
        )
        parameters = eqx.filter(model, eqx.is_inexact_array)
        updates, optimizer = self.transform.update(grads, optimizer, parameters)
        return eqx.apply_updates(model, updates), state, optimizer, loss

    def prepare(self, operation: Operation, *, compiled: bool):
        # Select and optionally compile the requested operation
        function = getattr(self, f"_{operation}")
        if compiled:
            function = eqx.filter_jit(function)

        if operation != "full_step":
            return lambda: function(self.model, self.state, self.batch)

        def step():
            self.model, self.state, self.optimizer, loss = function(
                self.model, self.state, self.optimizer, self.batch
            )
            return loss

        return step

    def synchronize(self, value: Any) -> None:
        jax.block_until_ready(value)

    def peak_memory(self) -> int | None:
        stats = jax.devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use") if stats else None

    def reset_peak_memory(self) -> None:
        """JAX does not expose an allocator peak reset."""
