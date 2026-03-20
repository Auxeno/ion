import functools

import jax
import jax.numpy as jnp
import pytest

from ion import nn


@pytest.fixture
def key():
    return jax.random.key(0)


def _build_layers(key):
    keys = iter(jax.random.split(key, 100))

    # Wrap Dropout so it matches the layer(x) calling convention used by generic tests.
    dropout = nn.Dropout(p=0.5, deterministic=True)
    dropout_wrapper = functools.partial(dropout, key=next(keys))

    # Wrap cells so they match the layer(x) calling convention.
    rnn_cell = nn.RNNCell(8, 16, key=next(keys))
    rnn_cell_wrapper = lambda x: rnn_cell(x, rnn_cell.initial_state)

    lstm_cell = nn.LSTMCell(8, 16, key=next(keys))
    lstm_cell_wrapper = lambda x: lstm_cell(x, lstm_cell.initial_state)[0]

    gru_cell = nn.GRUCell(8, 16, key=next(keys))
    gru_cell_wrapper = lambda x: gru_cell(x, gru_cell.initial_state)

    lru_cell = nn.LRUCell(8, 16, key=next(keys))
    lru_cell_wrapper = lambda x: lru_cell(x, lru_cell.initial_state)[0]

    s4d_cell = nn.S4DCell(8, 8, key=next(keys))
    s4d_cell_wrapper = lambda x: s4d_cell(x, s4d_cell.initial_state)[0]

    s5_cell = nn.S5Cell(8, 8, key=next(keys))
    s5_cell_wrapper = lambda x: s5_cell(x, s5_cell.initial_state)[0]

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
        (dropout_wrapper, jnp.ones((2, 8))),
        (nn.Embedding(16, 8, key=next(keys)), jnp.array([[0, 3, 7, 15], [1, 2, 5, 10]])),
        (rnn_cell_wrapper, jnp.ones((2, 8))),
        (lstm_cell_wrapper, jnp.ones((2, 8))),
        (gru_cell_wrapper, jnp.ones((2, 8))),
        (lambda x, _l=nn.RNN(8, 16, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
        (lambda x, _l=nn.LSTM(8, 16, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
        (lambda x, _l=nn.GRU(8, 16, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
        (lru_cell_wrapper, jnp.ones((2, 8))),
        (lambda x, _l=nn.LRU(8, 16, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
        (s4d_cell_wrapper, jnp.ones((2, 8))),
        (lambda x, _l=nn.S4D(8, 8, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
        (s5_cell_wrapper, jnp.ones((2, 8))),
        (lambda x, _l=nn.S5(8, 8, key=next(keys)): _l(x)[0], jnp.ones((2, 5, 8))),
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
    "rnn_cell",
    "lstm_cell",
    "gru_cell",
    "rnn",
    "lstm",
    "gru",
    "lru_cell",
    "lru",
    "s4d_cell",
    "s4d",
    "s5_cell",
    "s5",
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
    "rnn",
    "lstm",
    "gru",
    "lru",
    "s4d",
    "s5",
]


@pytest.fixture(params=_STRUCTURAL_LAYER_NAMES)
def structural_layer_and_input(request, key):
    layers = _build_layers(key)
    idx = _PARAM_NAMES.index(request.param)
    return layers[idx]
