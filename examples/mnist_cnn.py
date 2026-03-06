"""MNIST handwritten digit classification with a small CNN."""

import gzip
import struct
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import ion
import numpy as np
import optax
from jaxtyping import Array, Float, Int

from ion import nn
from tqdm import tqdm


def load_mnist() -> tuple[
    Float[Array, "60000 28 28 1"],
    Int[Array, " 60000"],
    Float[Array, "10000 28 28 1"],
    Int[Array, " 10000"],
]:
    """Download MNIST dataset and return (train_images, train_labels, test_images, test_labels)."""
    cache_dir = Path.home() / ".cache" / "ion" / "mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def fetch(filename: str) -> Path:
        path = cache_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url + filename, path)
        return path

    def parse_images(path: Path) -> Float[Array, "n 28 28 1"]:
        with gzip.open(path, "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return jnp.array(data.reshape(n, rows, cols, 1), dtype=jnp.float32) / 255.0

    def parse_labels(path: Path) -> Int[Array, " n"]:
        with gzip.open(path, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return jnp.array(data, dtype=jnp.int32)

    train_images = parse_images(fetch("train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(fetch("train-labels-idx1-ubyte.gz"))
    test_images = parse_images(fetch("t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(fetch("t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels


class CNN(nn.Module):
    conv_1: nn.Conv
    conv_2: nn.Conv
    pool: nn.MaxPool
    fc_1: nn.Linear
    fc_2: nn.Linear

    def __init__(self, *, key: jax.Array) -> None:
        keys = jax.random.split(key, 4)
        self.conv_1 = nn.Conv(2, 1, 16, kernel_size=3, padding=1, key=keys[0])
        self.conv_2 = nn.Conv(2, 16, 32, kernel_size=3, padding=1, key=keys[1])
        self.pool = nn.MaxPool(2, kernel_size=2)
        self.fc_1 = nn.Linear(32 * 7 * 7, 128, key=keys[2])
        self.fc_2 = nn.Linear(128, 10, key=keys[3])

    def __call__(self, x: Float[Array, "*b 28 28 1"]) -> Float[Array, "*b 10"]:
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
    updates, opt_state = optimizer.update(grads, opt_state, model.params)
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
    # Training hyperparameters
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
            images = train_images[batch_indices]
            labels = train_labels[batch_indices]

            # Update model and optimizer state
            model, opt_state, loss = train_step(model, opt_state, images, labels)

            epoch_loss += loss.item()

        # Model eval accuracy
        test_acc = accuracy(model, test_images, test_labels).item()
        print(f"  loss: {epoch_loss / num_batches:.4f}  test accuracy: {test_acc:.2%}")
