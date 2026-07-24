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

from ..module import Module


class Sequential(Module):
    """Chains single-argument layers in order, forwarding an optional key to layers that accept one.

    >>> model = Sequential(Linear(3, 16, key=keys[0]), Dropout(0.1), Linear(16, 1, key=keys[1]))
    >>> model(x, key=key)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Callable, ...]

    def __init__(self, *layers: Callable) -> None:

        for layer in layers:
            if not callable(layer):
                raise TypeError(f"Sequential expects callable layers, got {type(layer).__name__}")
        self.layers = layers

    def __call__(self, x: Any, *, key: PRNGKeyArray | None = None) -> Any:

        keys = [None] * len(self.layers) if key is None else jax.random.split(key, len(self.layers))

        for layer, key_layer in zip(self.layers, keys):
            if "key" in inspect.signature(layer).parameters:
                x = layer(x, key=key_layer)
            else:
                x = layer(x)

        return x

    def __getitem__(self, i: int | slice) -> "Callable | Sequential":
        if isinstance(i, slice):
            return Sequential(*self.layers[i])
        return self.layers[i]

    def __iter__(self) -> Iterator[Callable]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
