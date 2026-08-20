"""Vectorized model ensembles.

Modules:
    Ensemble  Builds and evaluates independently initialized modules together.

Members share one input and run over a leading ensemble axis. Outputs retain
that axis by default and can instead be reduced by their mean or sum.
"""

import inspect
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

from ...typing import PRNGKey
from ..module import Module


class Ensemble(Module):
    """Build and evaluate independently initialized modules together.

    >>> ensemble = Ensemble(lambda key: MLP([3, 16, 1], key=key), 4, key=key)
    >>> ensemble(x)  # (b, 3) -> (4, b, 1)
    """

    models: Module
    size: int
    reduction: Literal["mean", "sum"] | None

    def __init__(
        self,
        factory: Callable[[PRNGKey], Module],
        size: int,
        *,
        reduction: Literal["mean", "sum"] | None = None,
        key: PRNGKey,
    ) -> None:

        if size < 1:
            raise ValueError(f"size ({size}) must be at least 1")
        if reduction not in (None, "mean", "sum"):
            raise ValueError(f"unknown reduction {reduction!r}; expected None, 'mean', or 'sum'")

        models = jax.vmap(factory)(jax.random.split(key, size))
        if not isinstance(models, Module):
            raise TypeError(f"factory must return a Module, got {type(models).__name__}")

        self.models = models
        self.size = size
        self.reduction = reduction

    def __call__(
        self,
        x: Any,
        *args: Any,
        training: bool | None = None,
        key: PRNGKey | None = None,
        **kwargs: Any,
    ) -> Any:

        parameters = inspect.signature(self.models).parameters
        training_parameter = parameters.get("training")

        if training_parameter is not None:
            if training is None and training_parameter.default is inspect.Parameter.empty:
                raise ValueError(f"{type(self.models).__name__} requires training=True or False")
            if training is not None:
                kwargs["training"] = training

        if "key" not in parameters:
            x = jax.vmap(lambda model: model(x, *args, **kwargs))(self.models)
        elif key is None:
            x = jax.vmap(lambda model: model(x, *args, key=None, **kwargs))(self.models)
        else:
            keys = jax.random.split(key, self.size)
            x = jax.vmap(lambda model, model_key: model(x, *args, key=model_key, **kwargs))(
                self.models, keys
            )

        if self.reduction == "mean":
            x = jax.tree.map(lambda a: jnp.mean(a, axis=0), x)
        elif self.reduction == "sum":
            x = jax.tree.map(lambda a: jnp.sum(a, axis=0), x)

        return x
