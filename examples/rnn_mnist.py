"""Sequential MNIST classification with an LSTM.

Each 28x28 image is flattened to a length 784 sequence of single pixels,
making this a challenging long-range dependency task.
"""

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int
from tqdm import tqdm

import ion
from ion import nn

from ._common.datasets import load_mnist


class SeqModel(nn.Module):
    gru: nn.GRU
    fc: nn.Linear

    def __init__(self, hidden_dim: int = 128, *, key: jax.Array) -> None:
        keys = jax.random.split(key)
        self.gru = nn.GRU(1, hidden_dim, key=keys[0])
        self.fc = nn.Linear(hidden_dim, 10, key=keys[1])

    def __call__(self, x: Float[Array, "b 784 1"]) -> Float[Array, "b 10"]:
        _, h = self.gru(x)

        # Pass final GRU hidden state through dense layer
        logits = self.fc(h)

        return logits


def loss_fn(
    model: SeqModel,
    sequences: Float[Array, "b 784 1"],
    labels: Int[Array, " b"],
) -> Float[Array, ""]:
    """Compute mean cross-entropy loss over a batch."""
    logits = model(sequences)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss


@jax.jit
def train_step(
    model: SeqModel,
    optimizer: ion.Optimizer,
    sequences: Float[Array, "b 784 1"],
    labels: Int[Array, " b"],
) -> tuple[SeqModel, ion.Optimizer, Float[Array, ""]]:
    """Compute gradients, apply optimizer update, and return the new model state."""
    loss, grads = jax.value_and_grad(loss_fn)(model, sequences, labels)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss


@jax.jit
def accuracy(
    model: SeqModel,
    sequences: Float[Array, "n 784 1"],
    labels: Int[Array, " n"],
) -> Float[Array, ""]:
    """Compute classification accuracy over a batch."""
    logits = model(sequences)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return accuracy


if __name__ == "__main__":
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    NUM_EPOCHS = 10

    # Load full dataset into device memory
    train_images, train_labels, test_images, test_labels = load_mnist()
    num_batches = len(train_images) // BATCH_SIZE

    # Flatten to sequences: (n, 28, 28, 1) -> (n, 784, 1)
    train_seq = train_images.reshape(len(train_images), -1, 1)
    test_seq = jnp.asarray(test_images.reshape(len(test_images), -1, 1), dtype=jnp.float32) / 255.0
    test_lab = jnp.asarray(test_labels)

    model = SeqModel(key=jax.random.key(0))
    optimizer = ion.Optimizer(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE)), model
    )

    for epoch in range(NUM_EPOCHS):
        # Shuffle training indices each epoch
        key = jax.random.key(epoch)
        indices = jax.random.permutation(key, len(train_images))

        epoch_loss = 0.0
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            # Slice a batch from the shuffled indices
            batch_indices = indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            sequences = jnp.asarray(train_seq[batch_indices], dtype=jnp.float32) / 255.0
            labels = jnp.asarray(train_labels[batch_indices])

            # Update model and optimizer
            model, optimizer, loss = train_step(model, optimizer, sequences, labels)

            epoch_loss += loss.item()

        # Evaluate accuracy on test set
        test_acc = accuracy(model, test_seq, test_lab).item()
        print(f"  loss: {epoch_loss / num_batches:.4f}  test accuracy: {test_acc:.2%}")
