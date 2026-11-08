"""Save and load pytree data such as models and optimizer states.

Functions:
    save    Serialize array leaves and metadata to a .ion file.
    load    Load array leaves and metadata from a .ion file into a reference tree.

The .ion format is safetensors-style (length-prefixed JSON header, then raw
tensor bytes) with native support for JAX dtypes such as bfloat16, float8 and
complex64. Array leaves, Python scalar leaves and Param trainable flags are saved.
The tree structure, including module fields such as sizes and activations, comes
from the reference tree on load.
"""

import contextlib
import json
import math
import os
import struct
import tempfile
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from .nn.buffer import Buffer
from .nn.param import Param
from .tree import _is_param_or_buffer
from .typing import PyTree

_FORMAT_VERSION = 2
_MAX_HEADER_BYTES = 100_000_000

_DTYPES = {
    "BOOL": np.dtype(bool),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "U16": np.dtype(np.uint16),
    "I16": np.dtype(np.int16),
    "U32": np.dtype(np.uint32),
    "I32": np.dtype(np.int32),
    "U64": np.dtype(np.uint64),
    "I64": np.dtype(np.int64),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "C64": np.dtype(np.complex64),
}
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}


def _path_key(key_path: tuple) -> str:
    return "".join(str(k) for k in key_path).lstrip(".")


def save(path: str | os.PathLike[str], pytree: PyTree) -> None:
    """Serialize a pytree's array leaves and metadata to a `.ion` file.

    Parameters
    ----------
    path : str | os.PathLike
        Destination file path (`.ion` appended if missing).
    pytree : PyTree
        Pytree to serialize. Array leaves, `Buffer` values, Python scalar leaves and
        `Param` trainable flags are written.

    Examples
    --------
    >>> ion.checkpoint.save("model.ion", model)
    """
    leaves_with_paths = jax.tree.flatten_with_path(pytree, is_leaf=_is_param_or_buffer)[0]

    # Collect array leaves, scalar leaves and Param trainable flags, keyed by tree path
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, bool | int | float | str] = {}
    trainable_flags: dict[str, bool] = {}
    for key_path, leaf in leaves_with_paths:
        key = _path_key(key_path)
        if isinstance(leaf, Param):
            arrays[key] = np.asarray(leaf._value)
            trainable_flags[key] = leaf.trainable
        elif isinstance(leaf, Buffer):
            arrays[key] = np.asarray(leaf.value)
        elif isinstance(leaf, (jax.Array, np.ndarray)):
            # A PRNG key has no NumPy equivalent, so its raw key data is written instead
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                leaf = jax.random.key_data(leaf)
            arrays[key] = np.asarray(leaf)
        elif isinstance(leaf, (bool, int, float, str)):
            scalars[key] = leaf

    # Build the JSON header: dtype, shape and byte offsets per tensor, plus ion metadata
    metadata = {
        "format_version": str(_FORMAT_VERSION),
        "trainable": json.dumps(trainable_flags),
        "scalars": json.dumps(scalars),
    }
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for key, array in arrays.items():
        if array.dtype not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported dtype {array.dtype} for '{key}'")
        header[key] = {
            "dtype": _DTYPE_NAMES[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes

    # Space-pad the header so the data section starts at an 8-byte aligned offset
    header_bytes = json.dumps(header).encode()
    header_bytes += b" " * (-(8 + len(header_bytes)) % 8)

    # Write beside the destination, then atomically replace it
    path = os.fspath(path)
    if not path.endswith(".ion"):
        path += ".ion"
    fd, temporary_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for array in arrays.values():
                f.write(np.ascontiguousarray(array).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_path)


def load(path: str | os.PathLike[str], reference_pytree: PyTree) -> PyTree:
    """Load array leaves and metadata from a `.ion` file into a reference pytree.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a `.ion` file created by `save` (`.ion` appended if missing).
    reference_pytree : PyTree
        Provides the tree structure. Its array and Python scalar leaves are replaced.

    Returns
    -------
    PyTree
        Pytree with arrays, scalars and `Param` trainable flags restored from file.

    Notes
    -----
    `Buffer`s in the returned pytree are new, so the reference keeps its own state.

    Examples
    --------
    >>> model = ion.checkpoint.load("model.ion", model)
    """
    leaves_with_paths, tree_def = jax.tree.flatten_with_path(
        reference_pytree, is_leaf=_is_param_or_buffer
    )

    # Parse the 8-byte length prefix and JSON header
    path = os.fspath(path)
    if not path.endswith(".ion"):
        path += ".ion"
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError(f"Not a valid .ion file: '{path}'")
    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size > _MAX_HEADER_BYTES or 8 + header_size > len(data):
        raise ValueError(f"Header size ({header_size}) is invalid for '{path}'")
    header: dict[str, Any] = json.loads(data[8 : 8 + header_size])
    buffer = memoryview(data)[8 + header_size :]

    # Ion metadata: format version, Param trainable flags and scalar leaves
    metadata: dict[str, str] = header.pop("__metadata__", {})
    version = int(metadata.get("format_version", 0))
    if version > _FORMAT_VERSION:
        raise ValueError(f"Format version ({version}) is newer than supported ({_FORMAT_VERSION})")
    trainable_flags: dict[str, bool] = json.loads(metadata.get("trainable", "{}"))

    # Files saved before scalar leaves were written leave them to the reference
    scalars: dict[str, Any] | None = None
    if "scalars" in metadata:
        scalars = json.loads(metadata["scalars"])

    # Tensor entries must tile the data section exactly, with no gaps, overlaps or overruns
    offset = 0
    for key, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        if entry["dtype"] not in _DTYPES:
            raise ValueError(f"Unsupported dtype {entry['dtype']} for '{key}'")
        begin, end = entry["data_offsets"]
        nbytes = _DTYPES[entry["dtype"]].itemsize * math.prod(entry["shape"])
        if begin != offset or end - begin != nbytes:
            raise ValueError(f"Corrupt tensor offsets for '{key}'")
        offset = end
    if offset != len(buffer):
        raise ValueError(f"Data section ({len(buffer)} bytes) does not match header ({offset})")

    # Rebuild leaves from the reference, taking arrays, scalars and trainable flags from the file
    expected_keys: set[str] = set()
    missing_keys: list[str] = []
    shape_problems: list[str] = []
    loaded_leaves: list[Any] = []
    for key_path, leaf in leaves_with_paths:
        key = _path_key(key_path)
        if isinstance(leaf, (bool, int, float, str)) and scalars is not None:
            expected_keys.add(key)
            if key in scalars:
                loaded_leaves.append(scalars[key])
            else:
                missing_keys.append(key)
            continue
        # A reference built with jax.eval_shape holds ShapeDtypeStructs in place of arrays
        if not isinstance(leaf, (Param, Buffer, jax.Array, np.ndarray, jax.ShapeDtypeStruct)):
            loaded_leaves.append(leaf)
            continue
        if isinstance(leaf, Param):
            ref = leaf._value
        elif isinstance(leaf, Buffer):
            ref = leaf.value
        else:
            ref = leaf
        is_key = jnp.issubdtype(ref.dtype, jax.dtypes.prng_key)
        if is_key:
            # A typed PRNG key is stored as its raw data, so the reference is compared through it
            ref = jax.eval_shape(jax.random.key_data, ref)
        expected_keys.add(key)
        if key not in header:
            missing_keys.append(key)
            continue
        entry = header[key]
        begin, end = entry["data_offsets"]
        arr = np.frombuffer(buffer[begin:end], dtype=_DTYPES[entry["dtype"]])
        arr = arr.reshape(entry["shape"])
        if arr.shape != ref.shape:
            shape_problems.append(
                f"Shape mismatch for '{key}': saved {arr.shape}, reference {ref.shape}."
            )
            continue
        if arr.dtype != ref.dtype:
            warnings.warn(
                f"Dtype mismatch for '{key}': saved {arr.dtype} vs {ref.dtype}, using saved",
                stacklevel=2,
            )
        if isinstance(leaf, Param):
            trainable = trainable_flags.get(key, leaf.trainable)
            loaded_leaves.append(Param(jnp.array(arr), trainable=trainable))
        elif isinstance(leaf, Buffer):
            loaded_leaves.append(Buffer(jnp.array(arr)))
        elif is_key:
            # An abstract key cannot report its impl, but a placeholder key of the same dtype can
            impl = jax.random.key_impl(jnp.zeros((), leaf.dtype))
            loaded_leaves.append(jax.random.wrap_key_data(jnp.array(arr), impl=impl))
        else:
            loaded_leaves.append(jnp.array(arr))

    # Report every mismatch at once, so a single failed load shows the whole difference
    extra_keys = [key for key in (*header, *(scalars or ())) if key not in expected_keys]
    problems = []
    if missing_keys:
        problems.append(f"Structure mismatch: reference keys not in file: {missing_keys}.")
    if extra_keys:
        problems.append(f"Structure mismatch: file keys not in reference: {extra_keys}.")
    problems += shape_problems
    if problems:
        fix = "Build the reference with the same structure and sizes as the saved pytree."
        raise ValueError("\n".join([f"Cannot hydrate a reference from '{path}':", *problems, fix]))

    return tree_def.unflatten(loaded_leaves)
