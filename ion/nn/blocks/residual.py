"""Residual connections.

Modules:
    Residual  Adds a layer's output to its input.

The wrapped layer must preserve the input shape. Positional and keyword
arguments are forwarded unchanged.
"""

import inspect
from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module


class Residual(Module):
    """Add a layer's output to its input.

    >>> residual = Residual(Linear(16, 16, key=key))
    >>> residual(x)  # (*, 16) -> (*, 16)
    """

    layer: Callable[..., Array]

    def __init__(self, layer: Callable[..., Array]) -> None:

        self.layer = layer

    def __call__(
        self,
        x: Float[Array, "..."],
        *args: Any,
        training: bool | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs: Any,
    ) -> Float[Array, "..."]:

        parameters = inspect.signature(self.layer).parameters
        training_parameter = parameters.get("training")

        if training_parameter is not None:
            if training is None and training_parameter.default is inspect.Parameter.empty:
                raise ValueError(f"{type(self.layer).__name__} requires training=True or False")
            if training is not None:
                kwargs["training"] = training
        if "key" in parameters:
            kwargs["key"] = key

        x_layer = self.layer(x, *args, **kwargs)

        return x + x_layer
