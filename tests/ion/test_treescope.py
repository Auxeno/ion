"""Tests for Treescope formatter setup."""

import IPython
import numpy as np
import treescope
from IPython.core.formatters import HTMLFormatter
from jaxlib._jax import ArrayImpl

import ion


class _DisplayFormatter:
    def __init__(self):
        self.formatters = {"text/html": HTMLFormatter()}


class _IPython:
    def __init__(self):
        self.display_formatter = _DisplayFormatter()


def test_enable_treescope_no_ipython():
    """Enable treescope is a no-op outside IPython."""
    ion.enable_treescope()


def test_enable_treescope_everything_no_ipython():
    """Enable treescope with everything=True is a no-op outside IPython."""
    ion.enable_treescope(everything=True)


def test_disable_treescope_no_ipython():
    """Disable treescope is a no-op outside IPython."""
    ion.disable_treescope()


def test_enable_and_disable_treescope_formatters(monkeypatch):
    """Disable removes both concrete and all-types Treescope formatters."""
    ip = _IPython()
    html_fmt = ip.display_formatter.formatters["text/html"]
    rendered_types = (
        ion.nn.Module,
        ion.nn.Param,
        ion.nn.Buffer,
        ion.Optimizer,
        ArrayImpl,
        np.ndarray,
    )
    monkeypatch.setattr(IPython, "get_ipython", lambda: ip)

    ion.enable_treescope()

    assert all(rendered_type in html_fmt.type_printers for rendered_type in rendered_types)
    assert isinstance(treescope.active_autovisualizer.get(), treescope.ArrayAutovisualizer)

    html_fmt.for_type(object, lambda obj: treescope.render_to_html(obj))
    ion.disable_treescope()

    assert all(rendered_type not in html_fmt.type_printers for rendered_type in rendered_types)
    assert object not in html_fmt.type_printers
    assert treescope.active_autovisualizer.get() is None
