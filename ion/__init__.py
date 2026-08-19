"""A simple library for neural and graph networks in JAX."""

import sys
from importlib.metadata import version

from . import checkpoint as checkpoint
from . import gnn as gnn
from . import nn as nn
from . import tree as tree
from .checkpoint import load, save
from .cost import cost
from .optimizer import Optimizer
from .tree import astype, clone, freeze, is_buffer, is_param, is_trainable_param, unfreeze


def enable_statistics() -> None:
    """Describe parameter distributions when a model is echoed at an interactive prompt.

    >>> ion.enable_statistics()
    """
    # IPython imports itself into a session, so its absence means there is nothing to hook
    ipython = sys.modules.get("IPython")
    if ipython is None or ipython.get_ipython() is None:
        return

    # Only the echo path pays for the reductions, leaving repr cheap for logging
    def describe(model, printer, cycle):
        from . import _rendering

        printer.text(_rendering.module_repr(model, _rendering.statistics(model)))

    formatters = ipython.get_ipython().display_formatter.formatters
    formatters["text/plain"].for_type(nn.Module, describe)


enable_statistics()

__version__ = version("ion-nn")

__all__ = [
    "checkpoint",
    "gnn",
    "nn",
    "tree",
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
    "unfreeze",
]
