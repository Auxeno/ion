"""Bidirectional sequence processing.

Modules:
    Bidirectional  Runs sequence layers in both time directions.

Input layout is (batch, time, features). The backward outputs are returned to
the original time order before the two directions are combined. Sum is the
default mode and preserves the directional output dimension.
"""

from typing import Literal

import jax.numpy as jnp

from ...typing import Array, Float, PyTree
from ..module import Module


class Bidirectional(Module):
    """Run sequence layers in both time directions.

    >>> layer = Bidirectional(LSTM(3, 16, key=keys[0]), LSTM(3, 16, key=keys[1]))
    >>> outputs, states = layer(x)  # (b, t, 3) -> (b, t, 16), two final states
    """

    forward: Module
    backward: Module
    mode: Literal["sum", "concat", "mean"]

    def __init__(
        self,
        forward: Module,
        backward: Module,
        *,
        mode: Literal["sum", "concat", "mean"] = "sum",
    ) -> None:

        if mode not in ("sum", "concat", "mean"):
            raise ValueError(f"unknown mode {mode!r}; expected 'sum', 'concat', or 'mean'")

        self.forward = forward
        self.backward = backward
        self.mode = mode

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: tuple[PyTree, PyTree] | None = None,
    ) -> tuple[Float[Array, "b t o"], tuple[PyTree, PyTree]]:

        hx_forward, hx_backward = (None, None) if hx is None else hx

        x_forward, hx_forward = self.forward(x, hx_forward)
        x_backward, hx_backward = self.backward(jnp.flip(x, axis=1), hx_backward)
        x_backward = jnp.flip(x_backward, axis=1)

        if self.mode == "concat":
            x = jnp.concatenate((x_forward, x_backward), axis=-1)
        else:
            x = x_forward + x_backward
            if self.mode == "mean":
                x = x / 2

        return x, (hx_forward, hx_backward)
