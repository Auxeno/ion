"""Semi-supervised node classification on the Cora citation network.

The task is transductive: given a single graph of 2,708 academic papers
connected by 10,556 citation edges, predict which of 7 CS subfields each
paper belongs to. Only 140 labeled nodes are used for training; the rest
are unlabeled. Each node has a 1,433-dim bag-of-words feature vector.

Trains a GraphConv model (fixed degree-weighted averaging) and a
GraphAttention model (learned neighbor weighting) on the same split.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int
from tqdm import tqdm

import ion
from ion import gnn, nn

from ._common.datasets import load_cora


class GCNModel(nn.Module):
    graph_conv_1: gnn.GraphConv
    graph_conv_2: gnn.GraphConv

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key: jax.Array) -> None:
        key_1, key_2 = jax.random.split(key)
        self.graph_conv_1 = gnn.GraphConv(in_dim, hidden_dim, key=key_1)
        self.graph_conv_2 = gnn.GraphConv(hidden_dim, num_classes, key=key_2)

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
    graph_attention_1: gnn.GraphAttention
    graph_attention_2: gnn.GraphAttention

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key: jax.Array) -> None:
        key_1, key_2 = jax.random.split(key)
        self.graph_attention_1 = gnn.GraphAttention(in_dim, hidden_dim, num_heads=4, key=key_1)
        self.graph_attention_2 = gnn.GraphAttention(hidden_dim, num_classes, num_heads=1, key=key_2)

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

    # Train GraphConv model
    model = GCNModel(in_dim, HIDDEN_DIM, NUM_CLASSES, key=jax.random.key(0))
    optimizer = ion.Optimizer(optax.adam(LEARNING_RATE), model)

    for epoch in tqdm(range(NUM_EPOCHS), desc="GraphConv"):
        model, optimizer, loss = train_step(
            model, optimizer, x, senders, receivers, labels, train_mask
        )
        if epoch == NUM_EPOCHS - 1:
            test_acc = accuracy(model, x, senders, receivers, labels, test_mask).item()
            print(f"  loss: {loss.item():.4f}  test accuracy: {test_acc:.2%}")

    # Train GraphAttention model
    model = GATModel(in_dim, HIDDEN_DIM, NUM_CLASSES, key=jax.random.key(0))
    optimizer = ion.Optimizer(optax.adam(LEARNING_RATE), model)

    for epoch in tqdm(range(NUM_EPOCHS), desc="GraphAttention"):
        model, optimizer, loss = train_step(
            model, optimizer, x, senders, receivers, labels, train_mask
        )
        if epoch == NUM_EPOCHS - 1:
            test_acc = accuracy(model, x, senders, receivers, labels, test_mask).item()
            print(f"  loss: {loss.item():.4f}  test accuracy: {test_acc:.2%}")
