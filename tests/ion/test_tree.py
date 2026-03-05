import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from ion import nn, tree


class TestIsArray:
    def test_jax_array(self):
        assert tree.is_array(jnp.ones(3)) is True

    def test_numpy_array(self):
        assert tree.is_array(np.ones(3)) is True

    def test_param_is_not_array(self):
        assert tree.is_array(nn.Param(jnp.ones(3))) is False

    def test_scalar(self):
        assert tree.is_array(42) is False

    def test_none(self):
        assert tree.is_array(None) is False


class TestIsFloatArray:
    def test_float32(self):
        assert tree.is_float_array(jnp.array(1.0, dtype=jnp.float32)) is True

    def test_complex(self):
        assert tree.is_float_array(jnp.array(1 + 2j)) is True

    def test_int_array(self):
        assert tree.is_float_array(jnp.array(1)) is False

    def test_bool_array(self):
        assert tree.is_float_array(jnp.array(True)) is False

    def test_numpy_float(self):
        assert tree.is_float_array(np.array(1.0)) is True

    def test_numpy_int(self):
        assert tree.is_float_array(np.array(1)) is False

    def test_non_array(self):
        assert tree.is_float_array(1.0) is False


class TestFilter:
    def test_keeps_matching_replaces_rest(self):
        pytree = {"w": nn.Param(jnp.ones(2)), "s": 1.0}
        result = tree.filter(pytree, tree.is_param)
        assert isinstance(result["w"], nn.Param)
        assert result["s"] is None

    def test_nothing_matches(self):
        pytree = {"a": 1, "b": "hello"}
        result = tree.filter(pytree, tree.is_param)
        assert result["a"] is None
        assert result["b"] is None

    def test_on_module(self):
        class Model(nn.Module):
            w: nn.Param
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.scale = 2.0

        result = tree.filter(Model(), tree.is_param)
        assert isinstance(result.w, nn.Param)
        assert result.scale == 2.0

    def test_nested_dict(self):
        pytree = {"outer": {"w": nn.Param(jnp.ones(2)), "x": 5}}
        result = tree.filter(pytree, tree.is_param)
        assert isinstance(result["outer"]["w"], nn.Param)
        assert result["outer"]["x"] is None


class TestApplyUpdates:
    def test_param_update_preserves_wrapper(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        updates = {"w": nn.Param(jnp.array([0.1, 0.2]))}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is True
        npt.assert_allclose(result["w"].value, jnp.array([1.1, 2.2]))

    def test_frozen_param_unchanged(self):
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([9.0]))}
        result = tree.apply_updates(data, updates)
        assert result["w"].trainable is False
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_none_update_skipped(self):
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": None}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_plain_array_update(self):
        data = {"x": jnp.array([1.0, 2.0])}
        updates = {"x": jnp.array([0.5, 0.5])}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["x"], jnp.array([1.5, 2.5]))

    def test_raw_array_delta_on_param(self):
        """Update is a plain array (not Param), applied to Param.value."""
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": jnp.array([0.5])}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.5]))

    def test_on_module(self):
        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.b = nn.Param(jnp.zeros(2))

        model = Model()
        updates = Model()  # all ones/zeros as deltas
        result = tree.apply_updates(model, updates)
        npt.assert_allclose(result.w.value, jnp.array([2.0, 2.0]))
        npt.assert_allclose(result.b.value, jnp.zeros(2))


class TestSaveLoad:
    def test_roundtrip_on_module(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, model)
            loaded = tree.load(f.name, model)
        npt.assert_array_equal(loaded.w.value, model.w.value)

    def test_static_leaves_from_reference(self):
        """Non-array leaves come from the reference tree, not the file."""

        class Model(nn.Module):
            w: nn.Param
            count: int

            def __init__(self, key, count):
                self.w = nn.Param(jax.random.normal(key, (3,)))
                self.count = count

        original = Model(key=jax.random.key(0), count=5)
        reference = Model(key=jax.random.key(1), count=99)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, original)
            loaded = tree.load(f.name, reference)

        # Arrays come from the file (original)
        npt.assert_array_equal(loaded.w.value, original.w.value)
        # Static int comes from the reference tree
        assert loaded.count == 99

    def test_param_and_plain_array_mix(self):
        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.buf = jnp.array([10.0, 20.0])

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, model)
            loaded = tree.load(f.name, model)
        npt.assert_array_equal(loaded.w.value, model.w.value)
        npt.assert_array_equal(loaded.buf, model.buf)
