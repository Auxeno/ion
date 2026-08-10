"""Tests for Treescope formatter setup."""

import IPython
import jax
import numpy as np
import optax
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
    assert treescope.abbreviation_threshold.get() == 2

    html_fmt.for_type(object, lambda obj: treescope.render_to_html(obj))
    ion.disable_treescope()

    assert all(rendered_type not in html_fmt.type_printers for rendered_type in rendered_types)
    assert object not in html_fmt.type_printers
    assert treescope.active_autovisualizer.get() is None
    assert treescope.abbreviation_threshold.get() is None


class TestTreescopeRepr:
    def test_collapsed_child_closes_cleanly(self):
        """A collapsed submodule drops its last separator instead of trailing a comma."""

        class Container(ion.nn.Module):
            drop: ion.nn.Dropout

            def __init__(self):
                self.drop = ion.nn.Dropout(0.1)

        rendering = treescope.render_to_text(Container())
        assert "Dropout(p=0.1)" in rendering
        assert ", )" not in rendering

    def test_default_config_hidden_when_collapsed(self):
        """A collapsed layer shows only config differing from its constructor default."""

        class Container(ion.nn.Module):
            norm: ion.nn.LayerNorm

            def __init__(self):
                self.norm = ion.nn.LayerNorm(4)

        assert "eps=" not in treescope.render_to_text(Container())
        assert "eps=" in treescope.render_to_text(ion.nn.LayerNorm(4))

    def test_abbreviates_nested_modules_and_parameters(self):
        """Deeper nesting progressively removes module fields and parameter wrappers."""
        model = ion.nn.MLP([4, 8, 2], key=jax.random.key(0))

        with treescope.abbreviation_threshold.set_scoped(0):
            assert treescope.render_to_text(model) == "MLP(<58 params>)"
        with treescope.abbreviation_threshold.set_scoped(1):
            rendering = treescope.render_to_text(model)
            assert "Linear(w=float32(4, 8), b=float32(8,))" in rendering

    def test_frozen_params_and_buffers_remain_identifiable(self):
        """Compact rendering distinguishes frozen parameters and mutable buffers."""
        frozen = ion.nn.LayerNorm(4).freeze()
        with treescope.abbreviation_threshold.set_scoped(1):
            assert "float32(4,), frozen" in treescope.render_to_text(frozen)
            assert "Buffer(float32(4,))" in treescope.render_to_text(ion.nn.BatchNorm(4))

    def test_masked_optimizer_state_renders(self):
        """Partitioned state holds masked placeholders, which must not break rendering."""
        model = ion.nn.MLP([4, 8, 2], key=jax.random.key(0))
        model = model.at.layers[0].set(model.layers[0].freeze())
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert "Optimizer(" in treescope.render_to_text(optimizer)
