"""A simple library for neural and graph networks in JAX."""

from importlib.metadata import version

from . import checkpoint as checkpoint
from . import gnn as gnn
from . import nn as nn
from . import tree as tree
from . import typing as typing
from .checkpoint import load, save
from .cost import cost
from .optimizer import Optimizer
from .tree import astype, clone, freeze, is_buffer, is_param, is_trainable_param, unfreeze

# Set to False where a model repr is logged often enough for its device sync to matter
statistics = True

__version__ = version("ion-nn")

__all__ = [
    "checkpoint",
    "gnn",
    "nn",
    "tree",
    "typing",
    "Optimizer",
    "astype",
    "clone",
    "cost",
    "freeze",
    "is_buffer",
    "is_param",
    "is_trainable_param",
    "load",
    "save",
    "statistics",
    "unfreeze",
]
