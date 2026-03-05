"""Multi-layer perceptron blocks.

Modules:
    MLP  Fully connected network with configurable depth and activation.

He normal weight init, zeros for bias. Assumes ReLU activation.
No activation on the final layer by default.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..layers.linear import Linear
from ..module import Module


class MLP(Module):
    """Multi-layer perceptron with configurable hidden layers and activation.

    >>> mlp = MLP(3, 1, hidden_dim=64, num_hidden_layers=2, key=key)
    >>> mlp(x)  # (*, 3) -> (*, 1)
    """

    layers: tuple[Linear, ...]
    activation: Callable[[Array], Array]
    final_activation: Callable[[Array], Array]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: Callable[[Array], Array] = jax.nn.relu,
        final_activation: Callable[[Array], Array] = lambda x: x,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        assert num_hidden_layers >= 0, "MLP `num_hidden_layers` must be >= 0."

        if num_hidden_layers == 0:
            keys = jax.random.split(key, 1)
            layers = [Linear(in_dim, out_dim, bias, dtype, w_init, b_init, key=keys[0])]
        else:
            keys = jax.random.split(key, num_hidden_layers + 1)
            layers = []
            layers.append(Linear(in_dim, hidden_dim, bias, dtype, w_init, b_init, key=keys[0]))
            for i in range(num_hidden_layers - 1):
                layers.append(
                    Linear(hidden_dim, hidden_dim, bias, dtype, w_init, b_init, key=keys[i + 1])
                )
            layers.append(Linear(hidden_dim, out_dim, bias, dtype, w_init, b_init, key=keys[-1]))

        self.layers = tuple(layers)
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.final_activation(x)

        return x
