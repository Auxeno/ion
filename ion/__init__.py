"""Neural networks in JAX with immutable pytrees and explicit parameters."""

from . import nn as nn
from . import transforms as transforms
from . import tree as tree
from .transforms import (
    grad,
    value_and_grad,
)
from .tree import (
    apply_updates,
    freeze,
    is_param,
    is_trainable_param,
    load,
    save,
    unfreeze,
)


def enable_treescope() -> None:
    """Activate treescope as the default interactive renderer."""
    try:
        from treescope import basic_interactive_setup

        basic_interactive_setup()
    except Exception:
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
    except Exception:
        pass


# Enable Treescope by default
enable_treescope()

__all__ = [
    "nn",
    "tree",
    "apply_updates",
    "disable_treescope",
    "enable_treescope",
    "freeze",
    "grad",
    "is_param",
    "is_trainable_param",
    "load",
    "save",
    "unfreeze",
    "value_and_grad",
]
