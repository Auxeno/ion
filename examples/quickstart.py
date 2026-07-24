"""Build and train a small MLP with Ion and native JAX transformations."""

import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn


def loss_fn(model: nn.MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute mean squared error over a batch."""
    predictions = model(x)
    return jnp.mean((predictions - y) ** 2)


@jax.jit
def train_step(
    model: nn.MLP,
    optimizer: ion.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> tuple[nn.MLP, ion.Optimizer, jax.Array]:
    """Compute gradients, update the model, and return the loss."""
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss


if __name__ == "__main__":
    model = nn.MLP([2, 4, 1], key=jax.random.key(0))
    print(model)
    print(f"Parameters: {model.num_params}")

    x = jnp.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = jnp.array([
        [0.0],
        [-2.0],
        [1.0],
        [-1.0],
    ])

    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    optimizer = ion.Optimizer(optax.adam(1e-2), model)
    model, optimizer = optimizer.update(model, grads)

    for _ in range(500):
        model, optimizer, loss = train_step(model, optimizer, x, y)

    print(f"Loss: {float(loss):.2e}")
    print("Predictions:")
    print(jnp.round(model(x), 2))
