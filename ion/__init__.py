"""A simple library for neural and graph networks in JAX."""

from importlib.metadata import version

from . import checkpoint as checkpoint
from . import gnn as gnn
from . import nn as nn
from . import tree as tree
from .checkpoint import load, save
from .optimizer import Optimizer
from .tree import astype, clone, freeze, is_buffer, is_param, is_trainable_param, unfreeze


def enable_treescope(everything: bool = False) -> None:
    """Activate treescope as the default interactive renderer.

    >>> ion.enable_treescope()                 # Ion types and arrays only
    >>> ion.enable_treescope(everything=True)  # all types
    """
    try:
        import IPython  # type: ignore[reportMissingImports]

        ip = IPython.get_ipython()  # type: ignore[reportPrivateImportUsage]
        if ip is None:
            return

        if everything:
            from treescope import basic_interactive_setup

            basic_interactive_setup()
        else:
            import numpy as np
            import treescope
            from jaxlib._jax import ArrayImpl

            html_fmt = ip.display_formatter.formatters["text/html"]  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

            render = lambda obj: treescope.render_to_html(obj)

            html_fmt.for_type(nn.Module, render)
            html_fmt.for_type(nn.Param, render)
            html_fmt.for_type(nn.Buffer, render)
            html_fmt.for_type(Optimizer, render)
            html_fmt.for_type(ArrayImpl, render)
            html_fmt.for_type(np.ndarray, render)
            treescope.active_autovisualizer.set_globally(treescope.ArrayAutovisualizer())
            treescope.abbreviation_threshold.set_globally(2)
    except ImportError:
        pass


def enable_statistics() -> None:
    """Describe parameter distributions when a model is echoed at an interactive prompt.

    >>> ion.enable_statistics()
    """
    try:
        import IPython  # type: ignore[reportMissingImports]

        ip = IPython.get_ipython()  # type: ignore[reportPrivateImportUsage]
        if ip is None:
            return

        # Only the echo path pays for the reductions, leaving repr cheap for logging
        def describe(model, printer, cycle):
            from . import _rendering

            printer.text(_rendering.module_repr(model, _rendering.statistics(model)))

        text_fmt = ip.display_formatter.formatters["text/plain"]  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        text_fmt.for_type(nn.Module, describe)
    except ImportError:
        pass


def disable_treescope() -> None:
    """Deactivate all Treescope rendering.

    >>> ion.disable_treescope()
    """
    try:
        import IPython  # type: ignore[reportMissingImports]
        import numpy as np
        import treescope
        from jaxlib._jax import ArrayImpl

        ip = IPython.get_ipython()  # type: ignore[reportPrivateImportUsage]
        if ip is not None:
            html_fmt = ip.display_formatter.formatters["text/html"]  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            html_fmt.type_printers.pop(nn.Module, None)
            html_fmt.type_printers.pop(nn.Param, None)
            html_fmt.type_printers.pop(nn.Buffer, None)
            html_fmt.type_printers.pop(Optimizer, None)
            html_fmt.type_printers.pop(ArrayImpl, None)
            html_fmt.type_printers.pop(np.ndarray, None)
            html_fmt.type_printers.pop(object, None)

        treescope.active_autovisualizer.set_globally(None)
        treescope.abbreviation_threshold.set_globally(None)
    except ImportError:
        pass


# Enable Treescope by default
enable_treescope()
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
    "disable_treescope",
    "enable_treescope",
    "freeze",
    "is_buffer",
    "is_param",
    "is_trainable_param",
    "load",
    "save",
    "unfreeze",
]
