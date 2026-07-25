"""Identity layer.

Modules:
    Identity  Pass-through, returns input unchanged.
"""

from typing import Any

from ..module import Module


class Identity(Module):
    """Pass-through layer, ignores all arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return x
