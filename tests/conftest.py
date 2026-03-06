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

    # Wrap BatchNorm so it matches the layer(x) calling convention (extract output from tuple).
    bn = nn.BatchNorm(8)
    bn = bn.replace(state=bn.initial_state)
    bn_wrapper = lambda x: bn(x, training=False)[0]

    # Wrap cells so they match the layer(x) calling convention.
    lstm_cell = nn.LSTMCell(8, 16, key=next(keys))
    lstm_cell_wrapper = lambda x: lstm_cell(x, lstm_cell.initial_state)[0]

    gru_cell = nn.GRUCell(8, 16, key=next(keys))
    gru_cell_wrapper = lambda x: gru_cell(x, gru_cell.initial_state)

    return [
        (nn.Linear(8, 16, key=next(keys)), jnp.ones((8,))),
        (nn.Linear(8, 16, bias=False, key=next(keys)), jnp.ones((8,))),
        (nn.Conv(1, 3, 8, kernel_size=3, padding=1, key=next(keys)), jnp.ones((10, 3))),
        (nn.Conv(2, 3, 8, kernel_size=3, padding=1, key=next(keys)), jnp.ones((6, 6, 3))),
        (nn.SelfAttention(8, num_heads=2, key=next(keys)), jnp.ones((4, 8))),
        (nn.SelfAttention(8, num_heads=2, causal=True, key=next(keys)), jnp.ones((4, 8))),
        (nn.LayerNorm(8), jnp.ones((4, 8))),
        (nn.RMSNorm(8), jnp.ones((4, 8))),
        (nn.MaxPool(1, kernel_size=2), jnp.ones((10, 3))),
        (nn.MaxPool(2, kernel_size=2), jnp.ones((4, 4, 3))),
        (nn.AvgPool(1, kernel_size=2), jnp.ones((10, 3))),
        (nn.AvgPool(2, kernel_size=2), jnp.ones((4, 4, 3))),
        (nn.Identity(), jnp.ones((8,))),
        (nn.MLP(8, 16, 32, num_hidden_layers=2, key=next(keys)), jnp.ones((8,))),
        (dropout_wrapper, jnp.ones((8,))),
        (nn.Embedding(16, 8, key=next(keys)), jnp.array([0, 3, 7, 15])),
        (bn_wrapper, jnp.ones((4, 8))),
        (lstm_cell_wrapper, jnp.ones((8,))),
        (gru_cell_wrapper, jnp.ones((8,))),
        (lambda x, _l=nn.LSTM(8, 16, key=next(keys)): _l(x)[0], jnp.ones((5, 8))),
        (lambda x, _l=nn.GRU(8, 16, key=next(keys)): _l(x)[0], jnp.ones((5, 8))),
        (nn.Sequential(nn.Linear(8, 16, key=next(keys)), jax.nn.relu), jnp.ones((8,))),
        (nn.LoRALinear(nn.Linear(8, 16, key=next(keys)), rank=4, key=next(keys)), jnp.ones((8,))),
        (nn.ConvTranspose(1, 3, 8, kernel_size=3, padding=1, key=next(keys)), jnp.ones((10, 3))),
        (nn.ConvTranspose(2, 3, 8, kernel_size=3, padding=1, key=next(keys)), jnp.ones((6, 6, 3))),
        (nn.Upsample(1, scale_factor=2), jnp.ones((10, 3))),
        (nn.Upsample(2, scale_factor=2), jnp.ones((4, 4, 3))),
        (nn.LearnedPositionalEmbedding(16, 8, key=next(keys)), jnp.ones((10, 8))),
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
    "batchnorm",
    "lstm_cell",
    "gru_cell",
    "lstm",
    "gru",
    "sequential",
    "lora_linear",
    "conv_transpose_1d",
    "conv_transpose_2d",
    "upsample_1d",
    "upsample_2d",
    "learned_positional_embedding",
]


@pytest.fixture(params=_PARAM_NAMES)
def layer_and_input(request, key):
    layers = _build_layers(key)
    idx = _PARAM_NAMES.index(request.param)
    return layers[idx]
