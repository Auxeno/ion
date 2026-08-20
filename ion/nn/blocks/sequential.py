"""Sequential container.

Modules:
    Sequential  Chains layers, calling each in order.

Accepts any callable (modules, functions like `jax.nn.relu`, lambdas).
Supports indexing, slicing, and iteration.
Layers accepting a `key` kwarg receive a per-layer key.
"""

import inspect
from collections.abc import Callable, Iterator
from typing import Any

import jax

from ...typing import PRNGKey
from ..module import Module


class Sequential(Module):
    """Chain layers, routing optional training mode and random keys.

    >>> model = Sequential(Linear(3, 16, key=keys[0]), Dropout(0.1), Linear(16, 1, key=keys[1]))
    >>> model(x, training=True, key=key)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Callable, ...]

    def __init__(self, *layers: Callable) -> None:

        self.layers = layers

    def __call__(
        self,
        x: Any,
        *,
        training: bool | None = None,
        key: PRNGKey | None = None,
    ) -> Any:

        keys = [None] * len(self.layers) if key is None else jax.random.split(key, len(self.layers))

        for layer, layer_key in zip(self.layers, keys):
            parameters = inspect.signature(layer).parameters
            training_parameter = parameters.get("training")
            kwargs = {}

            if training_parameter is not None:
                if training is None and training_parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"{type(layer).__name__} requires training=True or False")
                if training is not None:
                    kwargs["training"] = training
            if "key" in parameters:
                kwargs["key"] = layer_key

            x = layer(x, **kwargs)

        return x

    def __getitem__(self, i: int | slice) -> "Callable | Sequential":
        if isinstance(i, slice):
            return Sequential(*self.layers[i])
        return self.layers[i]

    def __iter__(self) -> Iterator[Callable]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
