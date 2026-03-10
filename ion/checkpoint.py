"""Save and load pytree data such as models and optimizer states.

Functions:
    save    Serialize array leaves and metadata to .npz.
    load    Load array leaves and metadata from .npz into a reference tree.

Array leaves and Param trainable flags are saved. Non-array leaves (ints,
floats, callables) come from the reference tree on load.

See docs/internals.md for implementation details.
"""

import json
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from .nn.param import Param
from .tree import is_param

_METADATA_KEY = "__ion_metadata__"
_FORMAT_VERSION = 1


def _path_key(key_path: tuple) -> str:
    return "".join(str(k) for k in key_path).lstrip(".")


def save(path: str, pytree: PyTree) -> None:
    """Serialize a pytree's array leaves and metadata to a ``.npz`` file.

    Parameters
    ----------
    path : str
        Destination file path (``.npz`` appended if missing).
    pytree : PyTree
        Pytree to serialize. Only array leaves and ``Param`` trainable flags are written.

    Examples
    --------
    >>> ion.checkpoint.save("model.npz", model)
    """
    leaves_with_paths = jtu.tree_flatten_with_path(pytree, is_leaf=is_param)[0]

    arrays_to_save: dict[str, np.ndarray] = {}
    trainable_flags: dict[str, bool] = {}

    for key_path, leaf in leaves_with_paths:
        key = _path_key(key_path)
        if isinstance(leaf, Param):
            arrays_to_save[key + "._value"] = np.asarray(leaf._value)
            trainable_flags[key] = leaf.trainable
        elif isinstance(leaf, (jax.Array, np.ndarray)):
            arrays_to_save[key] = np.asarray(leaf)

    metadata = json.dumps(
        {"format_version": _FORMAT_VERSION, "trainable": trainable_flags}
    ).encode()
    arrays_to_save[_METADATA_KEY] = np.frombuffer(metadata, dtype=np.uint8).copy()

    np.savez(path, **arrays_to_save)  # type: ignore[reportArgumentType]


def load(path: str, reference_pytree: PyTree) -> PyTree:
    """Load array leaves and metadata from a ``.npz`` file into a reference pytree.

    Parameters
    ----------
    path : str
        Path to a ``.npz`` file created by :func:`save`.
    reference_pytree : PyTree
        Provides tree structure and non-array leaves; array leaves are replaced.

    Returns
    -------
    PyTree
        Pytree with arrays and ``Param`` trainable flags restored from file.

    Examples
    --------
    >>> model = ion.checkpoint.load("model.npz", model)
    """
    leaves_with_paths, tree_def = jtu.tree_flatten_with_path(reference_pytree, is_leaf=is_param)
    saved_data = np.load(path)

    metadata: dict[str, Any] = json.loads(saved_data[_METADATA_KEY].tobytes())
    trainable_flags: dict[str, bool] = metadata.get("trainable", {})
    array_keys_in_file = {k for k in saved_data.files if k != _METADATA_KEY}

    expected_keys: set[str] = set()
    loaded_leaves: list[Any] = []
    for key_path, leaf in leaves_with_paths:
        key = _path_key(key_path)
        if isinstance(leaf, Param):
            array_key = key + "._value"
            expected_keys.add(array_key)
            if array_key not in saved_data:
                raise ValueError(
                    f"Structure mismatch: reference tree expects key '{array_key}', "
                    f"but it was not found in the file. "
                    f"Available keys: {sorted(array_keys_in_file)}"
                )
            saved_array = saved_data[array_key]
            ref_shape = leaf._value.shape
            if saved_array.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch for '{array_key}': "
                    f"saved {saved_array.shape} vs reference {ref_shape}"
                )
            trainable = trainable_flags.get(key, leaf.trainable)
            loaded_leaves.append(Param(jnp.array(saved_array), trainable=trainable))
        elif isinstance(leaf, (jax.Array, np.ndarray)):
            expected_keys.add(key)
            if key not in saved_data:
                raise ValueError(
                    f"Structure mismatch: reference tree expects key '{key}', "
                    f"but it was not found in the file. "
                    f"Available keys: {sorted(array_keys_in_file)}"
                )
            saved_array = saved_data[key]
            ref_shape = leaf.shape
            if saved_array.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': "
                    f"saved {saved_array.shape} vs reference {ref_shape}"
                )
            loaded_leaves.append(jnp.array(saved_array))
        else:
            loaded_leaves.append(leaf)

    extra_keys = array_keys_in_file - expected_keys
    if extra_keys:
        raise ValueError(
            f"Structure mismatch: file contains keys not in reference tree: {sorted(extra_keys)}"
        )

    return tree_def.unflatten(loaded_leaves)
