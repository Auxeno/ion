"""Neural networks in JAX with immutable pytrees and explicit parameters."""

from . import checkpoint as checkpoint
from . import nn as nn
from . import tree as tree
from .checkpoint import (
    load,
    save,
)
from .tree import (
    apply_updates,
    freeze,
    is_param,
    is_trainable_param,
    unfreeze,
)


def enable_treescope() -> None:
    """Activate treescope as the default interactive renderer when inside an IPython env."""
    try:
        import IPython  # type: ignore[reportMissingImports]

        if IPython.get_ipython() is None:  # type: ignore[reportPrivateImportUsage]
            return

        from treescope import basic_interactive_setup

        basic_interactive_setup()
    except ImportError:
        pass


def disable_treescope() -> None:
    """Deactivate treescope as the default interactive renderer."""
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

__all__ = [
    "checkpoint",
    "nn",
    "tree",
    "apply_updates",
    "disable_treescope",
    "enable_treescope",
    "freeze",
    "is_param",
    "is_trainable_param",
    "load",
    "save",
    "unfreeze",
]
