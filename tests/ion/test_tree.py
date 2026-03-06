import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import nn, tree


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


class TestFreeze:
    def test_freeze_sets_trainable_false(self):
        data = {"w": nn.Param(jnp.ones(2)), "b": nn.Param(jnp.zeros(2))}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        assert frozen["b"].trainable is False

    def test_freeze_preserves_values(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        frozen = tree.freeze(data)
        npt.assert_array_equal(frozen["w"].value, jnp.array([1.0, 2.0]))

    def test_freeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        npt.assert_array_equal(frozen["w"].value, jnp.ones(2))

    def test_freeze_on_module(self):
        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3))

        m = Model()
        frozen = tree.freeze(m)
        assert frozen.w.trainable is False
        assert frozen.b.trainable is False

    def test_freeze_nested_module(self):
        class Inner(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))

        class Outer(nn.Module):
            inner: Inner
            b: nn.Param

            def __init__(self):
                self.inner = Inner()
                self.b = nn.Param(jnp.zeros(2))

        m = Outer()
        frozen = tree.freeze(m)
        assert frozen.inner.w.trainable is False
        assert frozen.b.trainable is False

    def test_freeze_does_not_affect_non_params(self):
        class Model(nn.Module):
            w: nn.Param
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.scale = 2.0

        m = Model()
        frozen = tree.freeze(m)
        assert frozen.scale == 2.0


class TestUnfreeze:
    def test_unfreeze_sets_trainable_true(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_unfreeze_preserves_values(self):
        data = {"w": nn.Param(jnp.array([3.0, 4.0]), trainable=False)}
        unfrozen = tree.unfreeze(data)
        npt.assert_array_equal(unfrozen["w"].value, jnp.array([3.0, 4.0]))

    def test_unfreeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=True)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_freeze_then_unfreeze_roundtrip(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        roundtripped = tree.unfreeze(tree.freeze(data))
        assert roundtripped["w"].trainable is True
        npt.assert_array_equal(roundtripped["w"].value, jnp.array([1.0, 2.0]))


class TestApplyUpdatesExtended:
    def test_apply_updates_preserves_trainable_flag(self):
        """After apply_updates, Param stays trainable."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=True)}
        updates = {"w": jnp.array([0.5])}
        result = tree.apply_updates(data, updates)
        assert result["w"].trainable is True

    def test_apply_updates_mixed_trainable_frozen(self):
        data = {
            "w": nn.Param(jnp.array([1.0, 2.0])),
            "b": nn.Param(jnp.array([0.0]), trainable=False),
        }
        updates = {
            "w": jnp.array([0.1, 0.2]),
            "b": jnp.array([9.0]),
        }
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"].value, jnp.array([1.1, 2.2]))
        npt.assert_allclose(result["b"].value, jnp.array([0.0]))  # unchanged


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

    def test_saved_keys_are_named(self):
        """Saved .npz keys use path names, not positional indices."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (4,)))
                self.b = nn.Param(jax.random.normal(k2, (2,)))

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, model)
            saved = np.load(f.name)
            assert sorted(saved.files) == ["b.value", "w.value"]

    def test_field_reorder_loads_correctly(self):
        """Reordering fields in the reference model still loads correctly."""

        class ModelV1(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (4,)))
                self.b = nn.Param(jax.random.normal(k2, (2,)))

        class ModelV2(nn.Module):
            b: nn.Param
            w: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.b = nn.Param(jax.random.normal(k2, (2,)))
                self.w = nn.Param(jax.random.normal(k1, (4,)))

        original = ModelV1(key=jax.random.key(0))
        reference = ModelV2(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, original)
            loaded = tree.load(f.name, reference)

        npt.assert_array_equal(loaded.w.value, original.w.value)
        npt.assert_array_equal(loaded.b.value, original.b.value)

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


class TestSaveLoadStructureMismatch:
    def test_load_fewer_saved_than_reference_raises(self):
        """Saving a smaller model and loading into a larger reference raises ValueError."""

        class SmallModel(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class BigModel(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (4,)))
                self.b = nn.Param(jax.random.normal(k2, (4,)))

        small = SmallModel(key=jax.random.key(0))
        big = BigModel(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, small)
            with pytest.raises(ValueError, match="Structure mismatch"):
                tree.load(f.name, big)

    def test_load_more_saved_than_reference_raises(self):
        """Saving a larger model and loading into smaller reference raises ValueError."""

        class SmallModel(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class BigModel(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (4,)))
                self.b = nn.Param(jax.random.normal(k2, (4,)))

        big = BigModel(key=jax.random.key(0))
        small = SmallModel(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, big)
            with pytest.raises(ValueError, match="Structure mismatch"):
                tree.load(f.name, small)

    def test_load_shape_mismatch_no_validation(self):
        """Loading arrays with mismatched shapes succeeds silently."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, dim):
                self.w = nn.Param(jax.random.normal(key, (dim,)))

        saved_model = Model(key=jax.random.key(0), dim=8)
        reference = Model(key=jax.random.key(1), dim=4)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, saved_model)
            loaded = tree.load(f.name, reference)
            assert loaded.w.shape == (8,)
            assert loaded.w.shape != reference.w.shape

    def test_save_load_trainable_flag_comes_from_reference(self):
        """The trainable flag is NOT saved — it comes from the reference tree."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, trainable=True):
                self.w = nn.Param(jax.random.normal(key, (4,)), trainable=trainable)

        frozen_model = Model(key=jax.random.key(0), trainable=False)
        assert frozen_model.w.trainable is False
        trainable_ref = Model(key=jax.random.key(1), trainable=True)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, frozen_model)
            loaded = tree.load(f.name, trainable_ref)

        npt.assert_array_equal(loaded.w.value, frozen_model.w.value)
        # trainable comes from reference, not the file
        assert loaded.w.trainable is True


class TestApplyUpdatesEdgeCases:
    def test_apply_updates_frozen_param_with_large_update_silently_ignored(self):
        """A large update to a frozen param is silently dropped — no warning."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([999999.0]))}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_apply_updates_update_trainable_flag_ignored(self):
        """The trainable flag on the update Param is ignored; the model's flag wins."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=True)}
        updates = {"w": nn.Param(jnp.array([0.5]), trainable=False)}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"].value, jnp.array([1.5]))
        assert result["w"].trainable is True
