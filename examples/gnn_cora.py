"""Semi-supervised node classification on the Cora citation network.

The task is transductive: given a single graph of 2,708 academic papers
connected by 10,556 citation edges, predict which of 7 CS subfields each
paper belongs to. Only 140 labeled nodes are used for training; the rest
are unlabeled. Each node has a 1,433-dim bag-of-words feature vector.

Trains a GCNConv model (fixed degree-weighted averaging) and a
GATConv model (learned neighbour weighting) on the same split.
"""

import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

import ion
from ion import gnn, nn
from ion.typing import Array, Bool, Float, Int


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


class GCNModel(nn.Module):
    graph_conv_1: gnn.GCNConv
    graph_conv_2: gnn.GCNConv

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key: jax.Array) -> None:
        key_1, key_2 = jax.random.split(key)
        self.graph_conv_1 = gnn.GCNConv(in_dim, hidden_dim, key=key_1)
        self.graph_conv_2 = gnn.GCNConv(hidden_dim, num_classes, key=key_2)

    def __call__(
        self,
        x: Float[Array, "n d"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n c"]:
        x = jax.nn.relu(self.graph_conv_1(x, senders, receivers))
        x = self.graph_conv_2(x, senders, receivers)
        return x


class GATModel(nn.Module):
    graph_attention_1: gnn.GATConv
    graph_attention_2: gnn.GATConv

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key: jax.Array) -> None:
        key_1, key_2 = jax.random.split(key)
        self.graph_attention_1 = gnn.GATConv(in_dim, hidden_dim, num_heads=4, key=key_1)
        self.graph_attention_2 = gnn.GATConv(hidden_dim, num_classes, num_heads=1, key=key_2)

    def __call__(
        self,
        x: Float[Array, "n d"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n c"]:
        x = jax.nn.elu(self.graph_attention_1(x, senders, receivers))
        x = self.graph_attention_2(x, senders, receivers)
        return x


def loss_fn(
    model: nn.Module,
    x: Float[Array, "n d"],
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    labels: Int[Array, " n"],
    mask: Bool[Array, " n"],
) -> Float[Array, ""]:
    """Compute masked cross-entropy loss over labeled nodes only."""
    logits = model(x, senders, receivers)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.where(mask, losses, 0.0).sum() / mask.sum()


@jax.jit
def train_step(
    model: nn.Module,
    optimizer: ion.Optimizer,
    x: Float[Array, "n d"],
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    labels: Int[Array, " n"],
    mask: Bool[Array, " n"],
) -> tuple[nn.Module, ion.Optimizer, Float[Array, ""]]:
    """Compute gradients on masked loss and apply optimizer update."""
    loss, grads = jax.value_and_grad(loss_fn)(model, x, senders, receivers, labels, mask)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss


@jax.jit
def accuracy(
    model: nn.Module,
    x: Float[Array, "n d"],
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    labels: Int[Array, " n"],
    mask: Bool[Array, " n"],
) -> Float[Array, ""]:
    """Compute classification accuracy over masked nodes."""
    logits = model(x, senders, receivers)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.where(mask, preds == labels, False).sum() / mask.sum()


if __name__ == "__main__":
    LEARNING_RATE = 5e-3
    NUM_EPOCHS = 10
    HIDDEN_DIM = 128
    NUM_CLASSES = 7

    # Load Cora dataset
    features, labels, senders, receivers, train_mask, val_mask, test_mask = load_cora()

    # Raw features are binary word counts, normalizing puts all nodes on the same scale
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    features = features / row_sums

    # Convert to JAX arrays
    x = jnp.asarray(features)  # (n, d) = (2708, 1433)
    labels = jnp.asarray(labels)  # (n,) = (2708,)
    senders = jnp.asarray(senders)  # (e,) = (10556,)
    receivers = jnp.asarray(receivers)  # (e,) = (10556,)
    train_mask = jnp.asarray(train_mask)  # (n,) 140 labeled nodes
    val_mask = jnp.asarray(val_mask)  # (n,) 500 nodes
    test_mask = jnp.asarray(test_mask)  # (n,) 1000 nodes

    # Add self-loops so nodes aggregate their own features
    num_nodes = x.shape[0]
    senders, receivers = gnn.add_self_loops(senders, receivers, num_nodes)

    in_dim = x.shape[1]

    # Train GCNConv model
    model = GCNModel(in_dim, HIDDEN_DIM, NUM_CLASSES, key=jax.random.key(0))
    optimizer = ion.Optimizer(optax.adam(LEARNING_RATE), model)

    for epoch in tqdm(range(NUM_EPOCHS), desc="GCNConv"):
        model, optimizer, loss = train_step(
            model, optimizer, x, senders, receivers, labels, train_mask
        )
        if epoch == NUM_EPOCHS - 1:
            test_acc = accuracy(model, x, senders, receivers, labels, test_mask).item()
            print(f"  loss: {loss.item():.4f}  test accuracy: {test_acc:.2%}")

    # Train GATConv model
    model = GATModel(in_dim, HIDDEN_DIM, NUM_CLASSES, key=jax.random.key(0))
    optimizer = ion.Optimizer(optax.adam(LEARNING_RATE), model)

    for epoch in tqdm(range(NUM_EPOCHS), desc="GATConv"):
        model, optimizer, loss = train_step(
            model, optimizer, x, senders, receivers, labels, train_mask
        )
        if epoch == NUM_EPOCHS - 1:
            test_acc = accuracy(model, x, senders, receivers, labels, test_mask).item()
            print(f"  loss: {loss.item():.4f}  test accuracy: {test_acc:.2%}")
