import urllib.request
from pathlib import Path

import numpy as np
from jaxtyping import Bool, UInt8


def load_mnist() -> tuple[
    UInt8[np.ndarray, "60000 28 28 1"],
    UInt8[np.ndarray, " 60000"],
    UInt8[np.ndarray, "10000 28 28 1"],
    UInt8[np.ndarray, " 10000"],
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

    # Add channels singleton dim to images
    train_images = data["x_train"][..., None]
    train_labels = data["y_train"]
    test_images = data["x_test"][..., None]
    test_labels = data["y_test"]

    return train_images, train_labels, test_images, test_labels


def load_dsprites() -> Bool[np.ndarray, "737280 64 64 1"]:
    """Download dSprites dataset and return binary images."""
    cache_dir = Path.home() / ".cache" / "ion" / "dsprites"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "dsprites.npz"

    if not path.exists():
        print("Downloading dsprites...")
        urllib.request.urlretrieve(
            "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
            "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            path,
        )

    data = np.load(path)

    # Add channels singleton dim and cast as bool (dsprites images are binary)
    images = data["imgs"][..., None].astype(bool)

    return images
