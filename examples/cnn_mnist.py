"""MNIST handwritten digit classification with a small CNN."""

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int
from tqdm import tqdm

import ion
from ion import nn

from ._common.datasets import load_mnist


class CNN(nn.Module):
    conv_1: nn.Conv
    conv_2: nn.Conv
    pool: nn.MaxPool
    fc_1: nn.Linear
    fc_2: nn.Linear

    def __init__(self, *, key: jax.Array) -> None:
        keys = jax.random.split(key, 4)
        self.conv_1 = nn.Conv(1, 16, kernel_shape=(3, 3), padding=1, key=keys[0])
        self.conv_2 = nn.Conv(16, 32, kernel_shape=(3, 3), padding=1, key=keys[1])
        self.pool = nn.MaxPool(kernel_shape=(2, 2))
        self.fc_1 = nn.Linear(32 * 7 * 7, 128, key=keys[2])
        self.fc_2 = nn.Linear(128, 10, key=keys[3])

    def __call__(self, x: Float[Array, "... 28 28 1"]) -> Float[Array, "... 10"]:
        x = self.conv_1(x)
        x = jax.nn.relu(x)
        x = self.pool(x)

        x = self.conv_2(x)
        x = jax.nn.relu(x)
        x = self.pool(x)

        x = x.reshape(*x.shape[:-3], -1)

        x = self.fc_1(x)
        x = jax.nn.relu(x)
        x = self.fc_2(x)
        return x


def loss_fn(
    model: CNN,
    images: Float[Array, "b 28 28 1"],
    labels: Int[Array, " b"],
) -> Float[Array, ""]:
    """Compute mean cross-entropy loss over a batch."""
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss


@jax.jit
def train_step(
    model: CNN,
    opt_state: optax.OptState,
    images: Float[Array, "b 28 28 1"],
    labels: Int[Array, " b"],
) -> tuple[CNN, optax.OptState, Float[Array, ""]]:
    """Compute gradients, apply optimizer update, and return the new model state."""
    loss, grads = ion.value_and_grad(loss_fn)(model, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = ion.apply_updates(model, updates)
    return model, opt_state, loss


@jax.jit
def accuracy(
    model: CNN,
    images: Float[Array, "n 28 28 1"],
    labels: Int[Array, " n"],
) -> Float[Array, ""]:
    """Compute classification accuracy over a batch."""
    logits = model(images)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return accuracy


if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    NUM_EPOCHS = 5

    optimizer = optax.adam(LEARNING_RATE)

    # Load full dataset into device memory
    train_images, train_labels, test_images, test_labels = load_mnist()
    num_batches = len(train_images) // BATCH_SIZE

    # Initialize model and optimizer state
    model = CNN(key=jax.random.key(0))
    opt_state = optimizer.init(model.params)

    for epoch in range(NUM_EPOCHS):
        # Shuffle training indices each epoch
        key = jax.random.key(epoch)
        indices = jax.random.permutation(key, len(train_images))

        epoch_loss = 0.0
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            # Slice a batch from the shuffled indices
            batch_indices = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            images = jnp.asarray(train_images[batch_indices], dtype=jnp.float32) / 255.0
            labels = jnp.asarray(train_labels[batch_indices])

            # Update model and optimizer state
            model, opt_state, loss = train_step(model, opt_state, images, labels)

            epoch_loss += loss.item()

        # Model eval accuracy
        test_acc = accuracy(
            model, jnp.asarray(test_images, dtype=jnp.float32) / 255.0, jnp.asarray(test_labels)
        ).item()
        print(f"  loss: {epoch_loss / num_batches:.4f}  test accuracy: {test_acc:.2%}")
