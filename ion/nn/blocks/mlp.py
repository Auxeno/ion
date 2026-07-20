"""Multi-layer perceptron blocks.

Modules:
    MLP  Fully connected network with configurable layer dimensions and activation.

He normal weight init, zeros for bias. Assumes ReLU activation.
No activation on the final layer by default.
"""

from collections.abc import Callable, Sequence

import jax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..layers.linear import Linear
from ..module import Module


class MLP(Module):
    """Multi-layer perceptron with configurable layer dimensions and activation.

    >>> mlp = MLP([3, 64, 64, 1], key=key)
    >>> mlp(x)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Linear, ...]
    activation: Callable[[Array], Array]
    final_activation: Callable[[Array], Array] | None

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Array], Array] = jax.nn.relu,
        final_activation: Callable[[Array], Array] | None = None,
        bias: bool = True,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if len(dims) < 2:
            raise ValueError(f"dims must have at least an input and output dim, got {list(dims)}")

        keys = jax.random.split(key, len(dims) - 1)
        self.layers = tuple(
            Linear(d_in, d_out, bias, w_init, b_init, key=layer_key)
            for d_in, d_out, layer_key in zip(dims[:-1], dims[1:], keys)
        )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Float[Array, "... i"]) -> Float[Array, "... o"]:

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
