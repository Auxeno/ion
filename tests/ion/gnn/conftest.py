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
    keys = iter(jax.random.split(key, 14))
    senders = jnp.array([0, 1, 1, 2, 0, 2, 0, 1, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0, 0, 1, 2])
    x = jax.random.normal(next(keys), (3, 8))
    return [
        (gnn.GCNConv(8, 16, key=next(keys)), x, senders, receivers),
        (gnn.GCNConv(8, 16, bias=False, key=next(keys)), x, senders, receivers),
        (gnn.GATConv(8, 16, num_heads=2, key=next(keys)), x, senders, receivers),
        (gnn.GATConv(8, 16, num_heads=4, key=next(keys)), x, senders, receivers),
        (gnn.GATConv(8, 16, num_heads=2, bias=False, key=next(keys)), x, senders, receivers),
        (gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=next(keys)), x, senders, receivers),
        (gnn.GATv2Conv(8, 16, num_heads=2, key=next(keys)), x, senders, receivers),
        (gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=next(keys)), x, senders, receivers),
    ]


_GNN_PARAM_NAMES = [
    "gcn_conv",
    "gcn_conv_no_bias",
    "gat_conv",
    "gat_conv_4_heads",
    "gat_conv_no_bias",
    "gat_conv_edge_dim",
    "gat_v2_conv",
    "gat_v2_conv_edge_dim",
]


@pytest.fixture(params=_GNN_PARAM_NAMES)
def gnn_layer_and_graph(request):
    layers = _build_gnn_layers(jax.random.key(0))
    idx = _GNN_PARAM_NAMES.index(request.param)
    return layers[idx]
