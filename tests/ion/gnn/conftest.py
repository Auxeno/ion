import jax
import jax.numpy as jnp
import pytest

from ion import gnn


@pytest.fixture
def triangle_graph():
    """Undirected triangle (3 nodes, 6 edges) with self-loops."""
    senders = jnp.array([0, 1, 1, 2, 0, 2, 0, 1, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0, 0, 1, 2])
    return senders, receivers


@pytest.fixture
def triangle_graph_no_self_loops():
    """Undirected triangle (3 nodes, 6 edges) without self-loops."""
    senders = jnp.array([0, 1, 1, 2, 0, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0])
    return senders, receivers


def _build_gnn_layers(key):
    keys = iter(jax.random.split(key, 10))
    senders = jnp.array([0, 1, 1, 2, 0, 2, 0, 1, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0, 0, 1, 2])
    x = jax.random.normal(next(keys), (3, 8))
    return [
        (gnn.GraphConv(8, 16, key=next(keys)), x, senders, receivers),
        (gnn.GraphConv(8, 16, bias=False, key=next(keys)), x, senders, receivers),
        (gnn.GraphAttention(8, 16, num_heads=2, key=next(keys)), x, senders, receivers),
        (gnn.GraphAttention(8, 16, num_heads=4, key=next(keys)), x, senders, receivers),
        (gnn.GraphAttention(8, 16, num_heads=2, bias=False, key=next(keys)), x, senders, receivers),
    ]


_GNN_PARAM_NAMES = [
    "graph_conv",
    "graph_conv_no_bias",
    "graph_attention",
    "graph_attention_4_heads",
    "graph_attention_no_bias",
]


@pytest.fixture(params=_GNN_PARAM_NAMES)
def gnn_layer_and_graph(request):
    layers = _build_gnn_layers(jax.random.key(0))
    idx = _GNN_PARAM_NAMES.index(request.param)
    return layers[idx]
