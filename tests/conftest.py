import jax
import jax.numpy as jnp
import pytest

from ion import nn


@pytest.fixture
def key():
    return jax.random.key(0)


def _build_layers(key):
    keys = iter(jax.random.split(key, 100))
    return [
        (nn.Linear(8, 16, key=next(keys)), jnp.ones((2, 8))),
        (nn.Linear(8, 16, bias=False, key=next(keys)), jnp.ones((2, 8))),
        (nn.Conv(3, 8, kernel_shape=(3,), padding=1, key=next(keys)), jnp.ones((2, 10, 3))),
        (nn.Conv(3, 8, kernel_shape=(3, 3), padding=1, key=next(keys)), jnp.ones((2, 6, 6, 3))),
        (nn.SelfAttention(8, num_heads=2, key=next(keys)), jnp.ones((2, 4, 8))),
        (nn.SelfAttention(8, num_heads=2, causal=True, key=next(keys)), jnp.ones((2, 4, 8))),
        (nn.LayerNorm(8), jnp.ones((2, 4, 8))),
        (nn.RMSNorm(8), jnp.ones((2, 4, 8))),
        (nn.MaxPool(kernel_shape=(2,)), jnp.ones((2, 10, 3))),
        (nn.MaxPool(kernel_shape=(2, 2)), jnp.ones((2, 4, 4, 3))),
        (nn.AvgPool(kernel_shape=(2,)), jnp.ones((2, 10, 3))),
        (nn.AvgPool(kernel_shape=(2, 2)), jnp.ones((2, 4, 4, 3))),
        (nn.Identity(), jnp.ones((2, 8))),
        (nn.MLP(8, 16, 32, num_hidden_layers=2, key=next(keys)), jnp.ones((2, 8))),
        (nn.Dropout(p=0.5, deterministic=True), jnp.ones((2, 8))),
        (nn.Embedding(16, 8, key=next(keys)), jnp.array([[0, 3, 7, 15], [1, 2, 5, 10]])),
        (nn.Sequential(nn.Linear(8, 16, key=next(keys)), jax.nn.relu), jnp.ones((2, 8))),
        (nn.LoRALinear(nn.Linear(8, 16, key=next(keys)), rank=4, key=next(keys)), jnp.ones((2, 8))),
        (
            nn.ConvTranspose(3, 8, kernel_shape=(3,), padding=1, key=next(keys)),
            jnp.ones((2, 10, 3)),
        ),
        (
            nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding=1, key=next(keys)),
            jnp.ones((2, 6, 6, 3)),
        ),
        (nn.LearnedPositionalEmbedding(16, 8, key=next(keys)), jnp.ones((2, 10, 8))),
    ]


_PARAM_NAMES = [
    "linear",
    "linear_no_bias",
    "conv_1d",
    "conv_2d",
    "attention",
    "attention_causal",
    "layernorm",
    "rmsnorm",
    "maxpool_1d",
    "maxpool_2d",
    "avgpool_1d",
    "avgpool_2d",
    "identity",
    "mlp",
    "dropout",
    "embedding",
    "sequential",
    "lora_linear",
    "conv_transpose_1d",
    "conv_transpose_2d",
    "learned_positional_embedding",
]


@pytest.fixture(params=_PARAM_NAMES)
def layer_and_input(request, key):
    layers = _build_layers(key)
    idx = _PARAM_NAMES.index(request.param)
    return layers[idx]


_STRUCTURAL_LAYER_NAMES = [
    "conv_1d",
    "conv_2d",
    "attention",
    "attention_causal",
    "maxpool_1d",
    "maxpool_2d",
    "avgpool_1d",
    "avgpool_2d",
    "conv_transpose_1d",
    "conv_transpose_2d",
]


@pytest.fixture(params=_STRUCTURAL_LAYER_NAMES)
def structural_layer_and_input(request, key):
    layers = _build_layers(key)
    idx = _PARAM_NAMES.index(request.param)
    return layers[idx]


def _build_seq_layers(key):
    keys = iter(jax.random.split(key, 10))
    return [
        (nn.RNN(8, 16, key=next(keys)), jnp.ones((2, 5, 8))),
        (nn.LSTM(8, 16, key=next(keys)), jnp.ones((2, 5, 8))),
        (nn.GRU(8, 16, key=next(keys)), jnp.ones((2, 5, 8))),
    ]


_SEQ_NAMES = ["rnn", "lstm", "gru"]


@pytest.fixture(params=_SEQ_NAMES)
def seq_layer_and_input(request, key):
    """Sequence layers (RNN, LSTM, GRU) with matching input."""
    layers = _build_seq_layers(key)
    idx = _SEQ_NAMES.index(request.param)
    return layers[idx]


def _build_cells(key):
    keys = iter(jax.random.split(key, 10))
    return [
        (nn.RNNCell(8, 16, key=next(keys)), jnp.ones((8,))),
        (nn.LSTMCell(8, 16, key=next(keys)), jnp.ones((8,))),
        (nn.GRUCell(8, 16, key=next(keys)), jnp.ones((8,))),
    ]


_CELL_NAMES = ["rnn_cell", "lstm_cell", "gru_cell"]


@pytest.fixture(params=_CELL_NAMES)
def cell_and_input(request, key):
    """RNN cells with matching input. Call as cell(x, cell.initial_state)."""
    cells = _build_cells(key)
    idx = _CELL_NAMES.index(request.param)
    return cells[idx]


def _build_ssm_layers(key):
    keys = iter(jax.random.split(key, 10))
    return [
        (nn.LRU(8, 16, key=next(keys)), jnp.ones((2, 5, 8))),
        (nn.S4D(8, 8, key=next(keys)), jnp.ones((2, 5, 8))),
        (nn.S5(8, 8, key=next(keys)), jnp.ones((2, 5, 8))),
    ]


_SSM_NAMES = ["lru", "s4d", "s5"]


@pytest.fixture(params=_SSM_NAMES)
def ssm_layer_and_input(request, key):
    """SSM layers (LRU, S4D, S5) with matching input."""
    layers = _build_ssm_layers(key)
    idx = _SSM_NAMES.index(request.param)
    return layers[idx]


def _build_ssm_cells(key):
    keys = iter(jax.random.split(key, 10))
    return [
        (nn.LRUCell(8, 16, key=next(keys)), jnp.ones((8,))),
        (nn.S4DCell(8, 8, key=next(keys)), jnp.ones((8,))),
        (nn.S5Cell(8, 8, key=next(keys)), jnp.ones((8,))),
    ]


_SSM_CELL_NAMES = ["lru_cell", "s4d_cell", "s5_cell"]


@pytest.fixture(params=_SSM_CELL_NAMES)
def ssm_cell_and_input(request, key):
    """SSM cells with matching input. Call as cell(x, cell.initial_state)."""
    cells = _build_ssm_cells(key)
    idx = _SSM_CELL_NAMES.index(request.param)
    return cells[idx]
