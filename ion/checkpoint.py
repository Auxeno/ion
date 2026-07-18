"""Save and load pytree data such as models and optimizer states.

Functions:
    save    Serialize array leaves and metadata to a .ion file.
    load    Load array leaves and metadata from a .ion file into a reference tree.

The .ion format is safetensors-style (length-prefixed JSON header, then raw
tensor bytes) with native support for JAX dtypes such as bfloat16, float8 and
complex64. Array leaves and Param trainable flags are saved; non-array leaves
(ints, floats, callables) come from the reference tree on load.
"""

import json
import math
import struct
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jaxtyping import PyTree

from .nn.param import Param
from .tree import is_param

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


def save(path: str, pytree: PyTree) -> None:
    """Serialize a pytree's array leaves and metadata to a `.ion` file.

    Parameters
    ----------
    path : str
        Destination file path (`.ion` appended if missing).
    pytree : PyTree
        Pytree to serialize. Only array leaves and `Param` trainable flags are written.

    Examples
    --------
    >>> ion.checkpoint.save("model.ion", model)
    """
    leaves_with_paths = jax.tree.flatten_with_path(pytree, is_leaf=is_param)[0]

    # Collect array leaves and Param trainable flags, keyed by tree path
    arrays: dict[str, np.ndarray] = {}
    trainable_flags: dict[str, bool] = {}
    for key_path, leaf in leaves_with_paths:
        key = _path_key(key_path)
        if isinstance(leaf, Param):
            arrays[key] = np.asarray(leaf._value)
            trainable_flags[key] = leaf.trainable
        elif isinstance(leaf, (jax.Array, np.ndarray)):
            arrays[key] = np.asarray(leaf)

    # Build the JSON header: dtype, shape and byte offsets per tensor, plus ion metadata
    metadata = {"format_version": str(_FORMAT_VERSION), "trainable": json.dumps(trainable_flags)}
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

    # Write the length prefix, header and raw tensor bytes
    if not path.endswith(".ion"):
        path += ".ion"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for array in arrays.values():
            f.write(np.ascontiguousarray(array).tobytes())


def load(path: str, reference_pytree: PyTree) -> PyTree:
    """Load array leaves and metadata from a `.ion` file into a reference pytree.

    Parameters
    ----------
    path : str
        Path to a `.ion` file created by `save` (`.ion` appended if missing).
    reference_pytree : PyTree
        Provides tree structure and non-array leaves; array leaves are replaced.

    Returns
    -------
    PyTree
        Pytree with arrays and `Param` trainable flags restored from file.

    Examples
    --------
    >>> model = ion.checkpoint.load("model.ion", model)
    """
    leaves_with_paths, tree_def = jax.tree.flatten_with_path(reference_pytree, is_leaf=is_param)

    # Parse the 8-byte length prefix and JSON header
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

    # Ion metadata: format version and Param trainable flags
    metadata: dict[str, str] = header.pop("__metadata__", {})
    version = int(metadata.get("format_version", 0))
    if version > _FORMAT_VERSION:
        raise ValueError(f"Format version ({version}) is newer than supported ({_FORMAT_VERSION})")
    trainable_flags: dict[str, bool] = json.loads(metadata.get("trainable", "{}"))

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

    # Rebuild leaves from the reference tree, taking arrays and trainable flags from the file
    expected_keys: set[str] = set()
    loaded_leaves: list[Any] = []
    for key_path, leaf in leaves_with_paths:
        if not isinstance(leaf, (Param, jax.Array, np.ndarray)):
            loaded_leaves.append(leaf)
            continue
        key = _path_key(key_path)
        ref = leaf._value if isinstance(leaf, Param) else leaf
        expected_keys.add(key)
        if key not in header:
            raise ValueError(f"Structure mismatch: reference key '{key}' not found in file")
        entry = header[key]
        begin, end = entry["data_offsets"]
        arr = np.frombuffer(buffer[begin:end], dtype=_DTYPES[entry["dtype"]])
        arr = arr.reshape(entry["shape"])
        if arr.shape != ref.shape:
            raise ValueError(f"Shape mismatch for '{key}': saved {arr.shape} vs {ref.shape}")
        if arr.dtype != ref.dtype:
            warnings.warn(
                f"Dtype mismatch for '{key}': saved {arr.dtype} vs {ref.dtype}, using saved",
                stacklevel=2,
            )
        if isinstance(leaf, Param):
            trainable = trainable_flags.get(key, leaf.trainable)
            loaded_leaves.append(Param(jnp.array(arr), trainable=trainable))
        else:
            loaded_leaves.append(jnp.array(arr))

    extra_keys = set(header) - expected_keys
    if extra_keys:
        raise ValueError(
            f"Structure mismatch: file contains keys not in reference tree: {sorted(extra_keys)}"
        )

    return tree_def.unflatten(loaded_leaves)
