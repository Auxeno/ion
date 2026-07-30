"""Equinox benchmark adapter."""

from importlib import metadata
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ...configs import ModelConfig
from ...protocol import Metric


class EquinoxWorkload:
    """Equinox implementation of the benchmark protocol."""

    framework_version = eqx.__version__
    software = {
        "equinox": framework_version,
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
    }

    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        module = __import__(
            f"benchmarks.implementations.equinox.{config.model}",
            fromlist=["create_model", "forward"],
        )
        key_model, key_inputs, key_targets = jax.random.split(jax.random.key(seed), 3)
        self.config = config
        self.module = module
        self.model = module.create_model(config, key=key_model)
        self.transform = optax.adamw(3e-4)
        parameters = eqx.filter(self.model, eqx.is_inexact_array)
        self.optimizer = self.transform.init(parameters)
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
        self.batch = (inputs, targets)
        self.parameter_count = sum(
            value.size for value in jax.tree.leaves(parameters) if value is not None
        )

    @staticmethod
    def _cast(model):
        return jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x,
            model,
        )

    def _forward(self, model, batch):
        inputs, _ = batch
        return self.module.forward(self._cast(model), inputs)

    def _loss(self, model, batch):
        _, targets = batch
        logits = self._forward(model, batch).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    def _forward_backward(self, model, batch):
        return eqx.filter_value_and_grad(self._loss)(model, batch)

    def _full_step(self, model, optimizer, batch):
        loss, grads = eqx.filter_value_and_grad(self._loss)(model, batch)
        parameters = eqx.filter(model, eqx.is_inexact_array)
        updates, optimizer = self.transform.update(grads, optimizer, parameters)
        return eqx.apply_updates(model, updates), optimizer, loss

    def prepare(self, metric: Metric, *, compiled: bool):
        if metric == "forward":
            function = self._forward
            if compiled:
                function = eqx.filter_jit(function)
            return lambda: function(self.model, self.batch)
        if metric == "forward_backward":
            function = self._forward_backward
            if compiled:
                function = eqx.filter_jit(function)
            return lambda: function(self.model, self.batch)

        function = self._full_step
        if compiled:
            function = eqx.filter_jit(function)

        def operation():
            self.model, self.optimizer, loss = function(self.model, self.optimizer, self.batch)
            return loss

        return operation

    def synchronize(self, value: Any) -> None:
        jax.block_until_ready(value)

    def peak_memory(self) -> int | None:
        stats = jax.devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use") if stats else None

    def reset_peak_memory(self) -> None:
        """JAX does not expose an allocator peak reset."""


def create_workload(config: ModelConfig, *, seed: int) -> EquinoxWorkload:
    """Create an Equinox benchmark workload."""
    return EquinoxWorkload(config, seed=seed)


def device_name() -> str:
    """Return the active JAX device name."""
    device = jax.devices()[0]
    return getattr(device, "device_kind", str(device))
