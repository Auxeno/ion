"""Tests for Treescope rendering."""

import jax
import optax
import treescope

import ion


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

    def test_collapsed_parameters_describe_shape_without_setup(self):
        """Nested parameters render as shapes with no Treescope configuration applied."""
        model = ion.nn.MLP([4, 8, 2], key=jax.random.key(0))

        rendering = treescope.render_to_text(model)
        assert "Linear(w=Param(float32(4, 8)), b=Param(float32(8,)))" in rendering

    def test_frozen_params_and_buffers_remain_identifiable(self):
        """Compact rendering distinguishes frozen parameters and mutable buffers."""
        assert "float32(4,), frozen" in treescope.render_to_text(ion.nn.LayerNorm(4).freeze())
        assert "Buffer(float32(4,))" in treescope.render_to_text(ion.nn.BatchNorm(4))

    def test_masked_optimizer_state_renders(self):
        """Partitioned state holds masked placeholders, which must not break rendering."""
        model = ion.nn.MLP([4, 8, 2], key=jax.random.key(0))
        model = model.at.layers[0].set(model.layers[0].freeze())
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert "Optimizer(" in treescope.render_to_text(optimizer)
