import json
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import checkpoint, nn, tree


class TestSaveLoad:
    def test_roundtrip_on_module(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, model)
        npt.assert_array_equal(loaded.w._value, model.w._value)

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
            checkpoint.save(f.name, original)
            loaded = checkpoint.load(f.name, reference)

        # Arrays come from the file (original)
        npt.assert_array_equal(loaded.w._value, original.w._value)
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
            checkpoint.save(f.name, model)
            saved = np.load(f.name)
            array_keys = sorted(k for k in saved.files if k != "__ion_metadata__")
            assert array_keys == ["b._value", "w._value"]

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
            checkpoint.save(f.name, original)
            loaded = checkpoint.load(f.name, reference)

        npt.assert_array_equal(loaded.w._value, original.w._value)
        npt.assert_array_equal(loaded.b._value, original.b._value)

    def test_param_and_plain_array_mix(self):
        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.buf = jnp.array([10.0, 20.0])

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, model)
        npt.assert_array_equal(loaded.w._value, model.w._value)
        npt.assert_array_equal(loaded.buf, model.buf)


class TestSaveLoadCallable:
    def test_callable_comes_from_reference_not_file(self):
        """Callable fields are restored from the reference tree, not the saved file."""
        from collections.abc import Callable

        class ModelWithAct(nn.Module):
            w: nn.Param
            act: Callable

            def __init__(self, act, *, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))
                self.act = act

        original = ModelWithAct(jax.nn.relu, key=jax.random.key(0))

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, original)
            ref = ModelWithAct(jax.nn.gelu, key=jax.random.key(1))
            loaded = checkpoint.load(f.name, ref)

        # Array data comes from the saved file
        npt.assert_array_equal(loaded.w._value, original.w._value)
        # Callable comes from the reference tree (gelu, not relu)
        assert loaded.act is jax.nn.gelu


class TestSaveLoadTrainable:
    def test_trainable_flag_roundtrip(self):
        """Trainable flags are saved and restored from the file."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (4,)), trainable=True)
                self.b = nn.Param(jax.random.normal(k2, (2,)), trainable=False)

        model = Model(key=jax.random.key(0))
        # Reference has opposite trainable flags
        reference = Model(key=jax.random.key(1))
        reference = reference.replace(
            w=nn.Param(reference.w._value, trainable=False),
            b=nn.Param(reference.b._value, trainable=True),
        )

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, reference)

        # Trainable flags come from the file, not the reference
        assert loaded.w.trainable is True
        assert loaded.b.trainable is False
        # Array values come from the file
        npt.assert_array_equal(loaded.w._value, model.w._value)
        npt.assert_array_equal(loaded.b._value, model.b._value)

    def test_frozen_model_save_restore(self):
        """A fully frozen model roundtrips with correct trainable=False flags."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = tree.freeze(Model(key=jax.random.key(0)))
        assert model.w.trainable is False

        # Reference is trainable (different from saved)
        reference = Model(key=jax.random.key(1))
        assert reference.w.trainable is True

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, reference)

        # Saved trainable=False wins over reference trainable=True
        assert loaded.w.trainable is False
        npt.assert_array_equal(loaded.w._value, model.w._value)

    def test_metadata_in_npz(self):
        """Saved .npz file contains __ion_metadata__ with trainable flags."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, trainable=True):
                self.w = nn.Param(jax.random.normal(key, (4,)), trainable=trainable)

        model = Model(key=jax.random.key(0), trainable=False)
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            saved = np.load(f.name)
            assert "__ion_metadata__" in saved.files

            metadata = json.loads(saved["__ion_metadata__"].tobytes())
            assert metadata["format_version"] == 1
            assert metadata["trainable"]["w"] is False

    def test_nested_module_trainable_flags(self):
        """Trainable flags roundtrip through nested modules."""

        class Inner(nn.Module):
            w: nn.Param

            def __init__(self, key, trainable=True):
                self.w = nn.Param(jax.random.normal(key, (2,)), trainable=trainable)

        class Outer(nn.Module):
            inner: Inner
            b: nn.Param

            def __init__(self, key, inner_trainable=True, b_trainable=True):
                k1, k2 = jax.random.split(key)
                self.inner = Inner(key=k1, trainable=inner_trainable)
                self.b = nn.Param(jax.random.normal(k2, (3,)), trainable=b_trainable)

        model = Outer(key=jax.random.key(0), inner_trainable=False, b_trainable=True)
        reference = Outer(key=jax.random.key(1), inner_trainable=True, b_trainable=True)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, reference)

        assert loaded.inner.w.trainable is False  # from file
        assert loaded.b.trainable is True  # from file
        npt.assert_array_equal(loaded.inner.w._value, model.inner.w._value)
        npt.assert_array_equal(loaded.b._value, model.b._value)


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
            checkpoint.save(f.name, small)
            with pytest.raises(ValueError, match="Structure mismatch"):
                checkpoint.load(f.name, big)

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
            checkpoint.save(f.name, big)
            with pytest.raises(ValueError, match="Structure mismatch"):
                checkpoint.load(f.name, small)

    def test_load_shape_mismatch_raises(self):
        """Loading arrays with mismatched shapes raises ValueError."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, dim):
                self.w = nn.Param(jax.random.normal(key, (dim,)))

        saved_model = Model(key=jax.random.key(0), dim=8)
        reference = Model(key=jax.random.key(1), dim=4)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, saved_model)
            with pytest.raises(ValueError, match="Shape mismatch"):
                checkpoint.load(f.name, reference)

    def test_load_plain_array_shape_mismatch_raises(self):
        """Loading plain arrays (non-Param) with mismatched shapes raises ValueError."""

        class Model(nn.Module):
            buf: jax.Array

            def __init__(self, dim):
                self.buf = jnp.zeros(dim)

        saved_model = Model(dim=8)
        reference = Model(dim=4)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, saved_model)
            with pytest.raises(ValueError, match="Shape mismatch"):
                checkpoint.load(f.name, reference)

    def test_load_dtype_mismatch_keeps_saved_dtype(self):
        """Loading arrays with mismatched dtypes succeeds, keeping the saved dtype."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, dtype):
                self.w = nn.Param(jax.random.normal(key, (4,)).astype(dtype))

        saved_model = Model(key=jax.random.key(0), dtype=jnp.float32)
        reference = Model(key=jax.random.key(1), dtype=jnp.float16)

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            checkpoint.save(f.name, saved_model)
            loaded = checkpoint.load(f.name, reference)
            assert loaded.w.dtype == jnp.float32
            assert loaded.w.dtype != reference.w.dtype
