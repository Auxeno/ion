import urllib.request
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int, UInt8


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


def load_cora() -> tuple[
    Float[np.ndarray, "2708 1433"],
    Int[np.ndarray, " 2708"],
    Int[np.ndarray, " 10556"],
    Int[np.ndarray, " 10556"],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Download Cora citation network and return features, labels, edges, and split masks.

    Returns (features, labels, senders, receivers, train_mask, val_mask, test_mask).
    Uses the standard Planetoid split: 140 train, 500 val, 1000 test nodes.
    """
    cache_dir = Path.home() / ".cache" / "ion" / "cora"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/"
    for filename in ("cora.content", "cora.cites"):
        path = cache_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)

    # Parse node features and labels from cora.content
    # Each line: <paper_id> <1433 binary features> <class_label>
    content = np.genfromtxt(cache_dir / "cora.content", dtype=str)
    paper_ids = content[:, 0].astype(int)
    features = content[:, 1:-1].astype(np.float32)
    label_names = content[:, -1]

    # Map paper IDs to contiguous node indices
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    # Encode string labels as integers
    classes = sorted(set(label_names))
    label_to_int = {name: i for i, name in enumerate(classes)}
    labels = np.array([label_to_int[name] for name in label_names], dtype=np.int32)

    # Parse edges from cora.cites and make undirected
    # Each line: <cited_id> <citing_id>
    cites = np.genfromtxt(cache_dir / "cora.cites", dtype=int)
    src = np.array([id_to_idx[pid] for pid in cites[:, 1]], dtype=np.int32)
    dst = np.array([id_to_idx[pid] for pid in cites[:, 0]], dtype=np.int32)
    senders = np.concatenate([src, dst])
    receivers = np.concatenate([dst, src])

    # Standard Planetoid split: 20 nodes per class for training
    num_nodes = len(paper_ids)
    rng = np.random.RandomState(42)
    train_indices = []
    for c in range(len(classes)):
        class_indices = np.where(labels == c)[0]
        train_indices.extend(rng.choice(class_indices, size=20, replace=False))
    train_indices = np.array(sorted(train_indices))

    remaining = np.setdiff1d(np.arange(num_nodes), train_indices)
    remaining = rng.permutation(remaining)
    val_indices = np.sort(remaining[:500])
    test_indices = np.sort(remaining[500:1500])

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return features, labels, senders, receivers, train_mask, val_mask, test_mask
