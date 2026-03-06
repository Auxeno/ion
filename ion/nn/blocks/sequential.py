"""Sequential container.

Modules:
    Sequential  Chains layers, calling each in order.

Accepts any callable (modules, functions like jax.nn.relu, lambdas).
Supports indexing, slicing, and iteration.
"""

from collections.abc import Iterator
from typing import Callable

from jaxtyping import Array

from ..module import Module


class Sequential(Module):
    """Container that chains layers, calling each in order.

    >>> model = Sequential(Linear(3, 16, key=k1), jax.nn.relu, Linear(16, 1, key=k2))
    >>> model(x)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Callable, ...]

    def __init__(self, *layers: Callable) -> None:
        for layer in layers:
            if not callable(layer):
                raise TypeError(f"Sequential expects callable layers, got {type(layer).__name__}")
        self.layers = layers

    def __call__(self, x: Array) -> Array:

        for layer in self.layers:
            x = layer(x)

        return x

    def __getitem__(self, i: int | slice) -> Callable | "Sequential":
        if isinstance(i, slice):
            return Sequential(*self.layers[i])
        return self.layers[i]

    def __iter__(self) -> Iterator[Callable]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)
