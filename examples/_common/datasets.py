import gzip
import struct
import urllib.request
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


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
