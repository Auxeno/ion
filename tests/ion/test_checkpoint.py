import json
import os
import struct
import tempfile
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import checkpoint, nn, tree


def read_header(path):
    """Parse the 8-byte length prefix and JSON header of an .ion file."""
    with open(path, "rb") as f:
        data = f.read()
    header_size = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8 : 8 + header_size])


class TestSaveLoad:
    def test_roundtrip_on_module(self):
        """Saving and loading a module preserves all parameter values."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, original)
            loaded = checkpoint.load(f.name, reference)

        # Arrays come from the file (original)
        npt.assert_array_equal(loaded.w._value, original.w._value)
        # Static int comes from the reference tree
        assert loaded.count == 99

    def test_saved_keys_are_named(self):
        """Saved tensor names are clean path names with no Param suffix."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (4,)))
                self.b = nn.Param(jax.random.normal(keys[1], (2,)))

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            header = read_header(f.name)
        array_keys = sorted(k for k in header if k != "__metadata__")
        assert array_keys == ["b", "w"]

    def test_buffer_keys_include_module_paths(self):
        """Buffer tensor names preserve the owning module path."""

        class Model(nn.Module):
            norm: nn.BatchNorm

            def __init__(self):
                self.norm = nn.BatchNorm(2)

        buffers = Model().init_buffers()
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, buffers)
            header = read_header(f.name)

        array_keys = sorted(k for k in header if k != "__metadata__")
        assert array_keys == ["norm[0]", "norm[1]"]

    def test_field_reorder_loads_correctly(self):
        """Reordering fields in the reference model still loads correctly."""

        class ModelV1(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (4,)))
                self.b = nn.Param(jax.random.normal(keys[1], (2,)))

        class ModelV2(nn.Module):
            b: nn.Param
            w: nn.Param

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.b = nn.Param(jax.random.normal(keys[1], (2,)))
                self.w = nn.Param(jax.random.normal(keys[0], (4,)))

        original = ModelV1(key=jax.random.key(0))
        reference = ModelV2(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, original)
            loaded = checkpoint.load(f.name, reference)

        npt.assert_array_equal(loaded.w._value, original.w._value)
        npt.assert_array_equal(loaded.b._value, original.b._value)

    def test_param_and_plain_array_mix(self):
        """Roundtrip preserves both Param and plain array leaves."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.buf = jnp.array([10.0, 20.0])

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, model)
        npt.assert_array_equal(loaded.w._value, model.w._value)
        npt.assert_array_equal(loaded.buf, model.buf)

    def test_path_without_extension(self):
        """Both save and load append .ion when the path lacks it."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))
        with tempfile.TemporaryDirectory() as d:
            checkpoint.save(os.path.join(d, "model"), model)
            assert os.path.exists(os.path.join(d, "model.ion"))
            loaded = checkpoint.load(os.path.join(d, "model"), model)
        npt.assert_array_equal(loaded.w._value, model.w._value)


class TestSaveLoadCallable:
    def test_callable_comes_from_reference_not_file(self):
        """Callable fields are restored from the reference tree, not the saved file."""

        class ModelWithAct(nn.Module):
            w: nn.Param
            act: Callable

            def __init__(self, act, *, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))
                self.act = act

        original = ModelWithAct(jax.nn.relu, key=jax.random.key(0))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (4,)), trainable=True)
                self.b = nn.Param(jax.random.normal(keys[1], (2,)), trainable=False)

        model = Model(key=jax.random.key(0))
        # Reference has opposite trainable flags
        reference = Model(key=jax.random.key(1))
        reference = reference.at.w.set(nn.Param(reference.w._value, trainable=False))
        reference = reference.at.b.set(nn.Param(reference.b._value, trainable=True))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, reference)

        # Saved trainable=False wins over reference trainable=True
        assert loaded.w.trainable is False
        npt.assert_array_equal(loaded.w._value, model.w._value)

    def test_metadata_in_header(self):
        """Saved file header contains __metadata__ with version and trainable flags."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, trainable=True):
                self.w = nn.Param(jax.random.normal(key, (4,)), trainable=trainable)

        model = Model(key=jax.random.key(0), trainable=False)
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            header = read_header(f.name)

        # Safetensors metadata values must be strings, so trainable flags are JSON-encoded
        metadata = header["__metadata__"]
        assert metadata["format_version"] == "2"
        assert json.loads(metadata["trainable"])["w"] is False

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
                keys = jax.random.split(key, 2)
                self.inner = Inner(key=keys[0], trainable=inner_trainable)
                self.b = nn.Param(jax.random.normal(keys[1], (3,)), trainable=b_trainable)

        model = Outer(key=jax.random.key(0), inner_trainable=False, b_trainable=True)
        reference = Outer(key=jax.random.key(1), inner_trainable=True, b_trainable=True)

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, reference)

        assert loaded.inner.w.trainable is False  # from file
        assert loaded.b.trainable is True  # from file
        npt.assert_array_equal(loaded.inner.w._value, model.inner.w._value)
        npt.assert_array_equal(loaded.b._value, model.b._value)


class TestDtypes:
    @pytest.mark.parametrize(
        "dtype",
        [
            jnp.bool_,
            jnp.uint8,
            jnp.uint16,
            jnp.uint32,
            jnp.int8,
            jnp.int16,
            jnp.int32,
            jnp.float16,
            jnp.bfloat16,
            jnp.float32,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.complex64,
        ],
    )
    def test_roundtrip(self, dtype):
        """Every supported dtype roundtrips with dtype and values intact."""
        original = {"x": jnp.arange(8).astype(dtype)}
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, original)
            loaded = checkpoint.load(f.name, {"x": jnp.zeros(8).astype(dtype)})
        assert loaded["x"].dtype == dtype
        npt.assert_array_equal(loaded["x"], original["x"])

    def test_header_dtype_names(self):
        """Header dtype strings follow the safetensors naming scheme."""
        pytree = {
            "a": jnp.ones(2, dtype=jnp.bfloat16),
            "b": jnp.ones(2, dtype=jnp.float8_e4m3fn),
            "c": jnp.ones(2, dtype=jnp.complex64),
            "d": np.ones(2, dtype=np.float64),
        }
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, pytree)
            header = read_header(f.name)
        assert header["['a']"]["dtype"] == "BF16"
        assert header["['b']"]["dtype"] == "F8_E4M3"
        assert header["['c']"]["dtype"] == "C64"
        assert header["['d']"]["dtype"] == "F64"

    def test_complex64_param_roundtrip(self):
        """Complex64 params (as in SSM layers) roundtrip with values intact."""

        class Model(nn.Module):
            A: nn.Param

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                re = jax.random.normal(keys[0], (4,))
                im = jax.random.normal(keys[1], (4,))
                self.A = nn.Param(re + 1j * im)

        model = Model(key=jax.random.key(0))
        assert model.A.dtype == jnp.complex64
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, model)
        assert loaded.A.dtype == jnp.complex64
        npt.assert_array_equal(loaded.A._value, model.A._value)

    def test_complex128_raises(self):
        """Saving a complex128 leaf raises ValueError (no C128 in the format)."""
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            with pytest.raises(ValueError, match="Unsupported dtype"):
                checkpoint.save(f.name, {"x": np.zeros(2, dtype=np.complex128)})

    def test_bfloat16_roundtrip_on_module(self):
        """bfloat16 params and plain arrays restore with dtype and values intact."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)).astype(jnp.bfloat16))
                self.buf = jnp.array([10.0, 20.0], dtype=jnp.bfloat16)

        model = Model(key=jax.random.key(0))
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            loaded = checkpoint.load(f.name, model)
        assert loaded.w.dtype == jnp.bfloat16
        assert loaded.buf.dtype == jnp.bfloat16
        npt.assert_array_equal(loaded.w._value, model.w._value)
        npt.assert_array_equal(loaded.buf, model.buf)


class TestFileFormat:
    def test_data_section_aligned(self):
        """The header is space-padded so the data section starts 8-byte aligned."""
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, {"x": jnp.ones(3)})
            with open(f.name, "rb") as saved:
                data = saved.read()
        header_size = struct.unpack("<Q", data[:8])[0]
        assert (8 + header_size) % 8 == 0

    def test_offsets_tile_data_section(self):
        """Tensor byte ranges are contiguous and exactly cover the data section."""
        pytree = {"a": jnp.ones((2, 3)), "b": jnp.ones(5, dtype=jnp.int32)}
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, pytree)
            with open(f.name, "rb") as saved:
                data = saved.read()
        header_size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + header_size])
        entries = [v for k, v in header.items() if k != "__metadata__"]
        entries.sort(key=lambda entry: entry["data_offsets"][0])
        offset = 0
        for entry in entries:
            assert entry["data_offsets"][0] == offset
            offset = entry["data_offsets"][1]
        assert offset == len(data) - 8 - header_size

    def test_truncated_file_raises(self):
        """A file shorter than the 8-byte length prefix raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(b"1234")
            f.flush()
            with pytest.raises(ValueError, match="Not a valid"):
                checkpoint.load(f.name, {})

    def test_oversized_header_raises(self):
        """A header size larger than the file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(struct.pack("<Q", 2**40) + b"{}")
            f.flush()
            with pytest.raises(ValueError, match="Header size"):
                checkpoint.load(f.name, {})

    def test_newer_format_version_raises(self):
        """A file with a newer format version raises ValueError."""
        header = json.dumps({"__metadata__": {"format_version": "99"}}).encode()
        header += b" " * (-(8 + len(header)) % 8)
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(struct.pack("<Q", len(header)) + header)
            f.flush()
            with pytest.raises(ValueError, match="Format version"):
                checkpoint.load(f.name, {})

    def test_corrupt_offsets_raise(self):
        """Tensor offsets that do not start at the previous end raise ValueError."""
        entry = {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}
        header = json.dumps({"x": entry}).encode()
        header += b" " * (-(8 + len(header)) % 8)
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(struct.pack("<Q", len(header)) + header + b"\x00" * 12)
            f.flush()
            with pytest.raises(ValueError, match="Corrupt tensor offsets"):
                checkpoint.load(f.name, {"x": jnp.zeros(2)})

    def test_data_size_mismatch_raises(self):
        """A data section that does not match the header's extent raises ValueError."""
        entry = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        header = json.dumps({"x": entry}).encode()
        header += b" " * (-(8 + len(header)) % 8)
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(struct.pack("<Q", len(header)) + header + b"\x00" * 12)
            f.flush()
            with pytest.raises(ValueError, match="Data section"):
                checkpoint.load(f.name, {"x": jnp.zeros(2)})

    def test_plain_safetensors_file_loads(self):
        """A minimal safetensors file with no __metadata__ loads with reference flags."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))
        target = np.arange(4, dtype=np.float32)
        entry = {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}
        header = json.dumps({"w": entry}).encode()
        header += b" " * (-(8 + len(header)) % 8)
        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            f.write(struct.pack("<Q", len(header)) + header + target.tobytes())
            f.flush()
            loaded = checkpoint.load(f.name, model)
        npt.assert_array_equal(loaded.w._value, target)
        assert loaded.w.trainable is True


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
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (4,)))
                self.b = nn.Param(jax.random.normal(keys[1], (4,)))

        small = SmallModel(key=jax.random.key(0))
        big = BigModel(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (4,)))
                self.b = nn.Param(jax.random.normal(keys[1], (4,)))

        big = BigModel(key=jax.random.key(0))
        small = SmallModel(key=jax.random.key(1))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
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

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, saved_model)
            with pytest.raises(ValueError, match="Shape mismatch"):
                checkpoint.load(f.name, reference)

    def test_load_dtype_mismatch_warns_and_keeps_saved_dtype(self):
        """Loading arrays with mismatched dtypes warns and keeps the saved dtype."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key, dtype):
                self.w = nn.Param(jax.random.normal(key, (4,)).astype(dtype))

        saved_model = Model(key=jax.random.key(0), dtype=jnp.float32)
        reference = Model(key=jax.random.key(1), dtype=jnp.float16)

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, saved_model)
            with pytest.warns(UserWarning, match="Dtype mismatch"):
                loaded = checkpoint.load(f.name, reference)
            assert loaded.w.dtype == jnp.float32
            assert loaded.w.dtype != reference.w.dtype

    def test_load_plain_array_dtype_mismatch_warns(self):
        """Loading plain arrays with mismatched dtypes warns."""

        class Model(nn.Module):
            buf: jax.Array

            def __init__(self, dtype):
                self.buf = jnp.zeros(4, dtype=dtype)

        saved_model = Model(dtype=jnp.float32)
        reference = Model(dtype=jnp.bfloat16)

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, saved_model)
            with pytest.warns(UserWarning, match="Dtype mismatch"):
                loaded = checkpoint.load(f.name, reference)
            assert loaded.buf.dtype == jnp.float32

    def test_load_matching_dtype_no_warning(self):
        """Loading with matching dtypes produces no warning."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(0))

        with tempfile.NamedTemporaryFile(suffix=".ion") as f:
            checkpoint.save(f.name, model)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loaded = checkpoint.load(f.name, model)
            npt.assert_array_equal(loaded.w._value, model.w._value)
