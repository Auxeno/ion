import urllib.request
from pathlib import Path

import numpy as np
from jaxtyping import Int, UInt8


def load_mnist() -> tuple[
    UInt8[np.ndarray, "60000 28 28 1"],
    Int[np.ndarray, " 60000"],
    UInt8[np.ndarray, "10000 28 28 1"],
    Int[np.ndarray, " 10000"],
]:
    """Download MNIST dataset and return (train_images, train_labels, test_images, test_labels)."""
    cache_dir = Path.home() / ".cache" / "ion" / "mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "mnist.npz"

    if not path.exists():
        print("Downloading MNIST...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            path,
        )

    data = np.load(path)
    train_images = data["x_train"][..., None].astype(np.uint8)
    train_labels = data["y_train"].astype(np.int32)
    test_images = data["x_test"][..., None].astype(np.uint8)
    test_labels = data["y_test"].astype(np.int32)
    return train_images, train_labels, test_images, test_labels
