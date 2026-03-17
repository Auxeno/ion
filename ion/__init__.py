"""Neural networks in JAX with immutable pytrees and explicit parameters."""

from importlib.metadata import version

from . import checkpoint as checkpoint
from . import nn as nn
from . import tree as tree
from .checkpoint import (
    load,
    save,
)
from .optimizer import Optimizer
from .tree import (
    astype,
    freeze,
    is_param,
    is_trainable_param,
    unfreeze,
)


def enable_treescope(everything: bool = False) -> None:
    """Activate treescope as the default interactive renderer.

    >>> ion.enable_treescope()                 # Ion Modules and Params only
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
            import treescope

            html_fmt = ip.display_formatter.formatters["text/html"]  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            render = treescope.render_to_html
            html_fmt.for_type(nn.Module, lambda obj: render(obj))
            html_fmt.for_type(nn.Param, lambda obj: render(obj))
            html_fmt.for_type(Optimizer, lambda obj: render(obj))
    except ImportError:
        pass


def disable_treescope() -> None:
    """Deactivate treescope rendering.

    >>> ion.disable_treescope()
    """
    try:
        import IPython  # type: ignore[reportMissingImports]
        import treescope

        ip = IPython.get_ipython()  # type: ignore[reportPrivateImportUsage]
        if ip is not None:
            html_fmt = ip.display_formatter.formatters["text/html"]  # type: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            html_fmt.type_printers.pop(object, None)

        treescope.active_autovisualizer.set_globally(None)
    except ImportError:
        pass


# Enable Treescope by default
enable_treescope()

__version__ = version("ion-nn")

__all__ = [
    "checkpoint",
    "nn",
    "tree",
    "Optimizer",
    "astype",
    "disable_treescope",
    "enable_treescope",
    "freeze",
    "is_param",
    "is_trainable_param",
    "load",
    "save",
    "unfreeze",
]
