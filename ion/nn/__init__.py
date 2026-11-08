"""Neural network modules: base classes, layers, and blocks."""

from typing import TYPE_CHECKING

from .layers.linear import Linear, Identity
from .layers.recurrent import GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell
from .layers.conv import Conv, ConvTranspose
from .layers.ssm import S4D, S4DCell, S5, S5Cell
from .layers.norm import LayerNorm, RMSNorm, BatchNorm, GroupNorm, SpectralNorm
from .layers.attention import MultiHeadAttention
from .blocks.mlp import MLP
from .layers.pool import AvgPool, MaxPool
from .layers.stochastic import Dropout, DropPath
from .blocks.bidirectional import Bidirectional
from .layers.embedding import Embedding
from .blocks.residual import Residual
from .blocks.ensemble import Ensemble
from .blocks.sequential import Sequential
from .layers.positional import LearnedPositionalEmbedding, RoPE, SinusoidalPositionalEmbedding
from .module import Module
from .param import Param
from .buffer import Buffer

__all__ = [
    "Module",
    "Buffer",
    "Param",
    "AvgPool",
    "BatchNorm",
    "Bidirectional",
    "Conv",
    "ConvTranspose",
    "Dropout",
    "DropPath",
    "Embedding",
    "Ensemble",
    "GroupNorm",
    "GRU",
    "GRUCell",
    "Identity",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "Linear",
    "LSTM",
    "LSTMCell",
    "MaxPool",
    "MLP",
    "MultiHeadAttention",
    "RMSNorm",
    "Residual",
    "RNN",
    "RNNCell",
    "RoPE",
    "S4D",
    "S4DCell",
    "S5",
    "S5Cell",
    "Sequential",
    "SinusoidalPositionalEmbedding",
    "SpectralNorm",
]

# Hidden from type checkers so unknown names stay static errors
if not TYPE_CHECKING:

    def __getattr__(name):
        import jax.nn

        jax_names = {n.replace("_", "").lower(): n for n in dir(jax.nn) if not n.startswith("_")}
        jax_name = jax_names.get(name.replace("_", "").lower())
        if jax_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        raise AttributeError(f"{name} is provided by JAX, use jax.nn.{jax_name}")
