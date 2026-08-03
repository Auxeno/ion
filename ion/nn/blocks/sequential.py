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
from jaxtyping import PRNGKeyArray

from ..buffer import Buffers
from ..module import Module


class Sequential(Module):
    """Chain layers, routing optional training mode, buffers, and random keys.

    >>> model = Sequential(Linear(3, 16, key=keys[0]), Dropout(0.1), Linear(16, 1, key=keys[1]))
    >>> model(x, training=True, key=key)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Callable, ...]

    def __init__(self, *layers: Callable) -> None:

        for layer in layers:
            if not callable(layer):
                raise TypeError(f"Sequential expects callable layers, got {type(layer).__name__}")
        self.layers = layers

    def __call__(
        self,
        x: Any,
        buffers: Buffers | None = None,
        *,
        training: bool | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Any:

        keys = [None] * len(self.layers) if key is None else jax.random.split(key, len(self.layers))
        return_buffers = buffers is not None

        for layer, layer_key in zip(self.layers, keys):
            parameters = inspect.signature(layer).parameters
            kwargs = {}

            if "training" in parameters:
                if training is None:
                    raise ValueError(f"{type(layer).__name__} requires training=True or False")
                kwargs["training"] = training
            if "key" in parameters:
                kwargs["key"] = layer_key
            if "buffers" in parameters:
                if buffers is None:
                    raise ValueError(f"{type(layer).__name__} requires model.init_buffers()")
                x, buffers = layer(x, buffers, **kwargs)
            else:
                x = layer(x, **kwargs)

        return (x, buffers) if return_buffers else x

    def __getitem__(self, i: int | slice) -> "Callable | Sequential":
        if isinstance(i, slice):
            return Sequential(*self.layers[i])
        return self.layers[i]

    def __iter__(self) -> Iterator[Callable]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
