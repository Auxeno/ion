"""Predicate-based utilities for filtering and serializing JAX pytrees.

Functions:
    is_param            Check if a leaf is a Param.
    is_trainable_param  Check if a leaf is a trainable Param.
    freeze              Set all Params to trainable=False.
    unfreeze            Set all Params to trainable=True.
    apply_updates       Add optimizer deltas to trainable parameters.
    save                Serialize array leaves to .npz.
    load                Load array leaves from .npz into a reference tree.

Classes:
    Static              Wraps a value so JAX treats it as static metadata.

Neural network pytrees mix `Param` wrappers, plain arrays, and static Python
values. Standard `jax.tree_util` treats every leaf uniformly, these utilities
provide selective filtering by type and trainability.

See docs/internals.md for implementation details.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from .nn.param import Param


def is_param(x: Any) -> bool:
    """Check if an object is a `Param`."""
    return isinstance(x, Param)


def is_trainable_param(x: Any) -> bool:
    """Check if an object is a trainable `Param`."""
    return isinstance(x, Param) and x.trainable


def freeze(pytree: PyTree) -> PyTree:
    """Return a copy with all `Param`s set to `trainable=False`.

    >>> frozen_model = ion.tree.freeze(model)
    """

    def _freeze_leaf(leaf):
        if isinstance(leaf, Param) and leaf.trainable:
            return Param(leaf.value, trainable=False)
        return leaf

    return jax.tree.map(_freeze_leaf, pytree, is_leaf=is_param)


def unfreeze(pytree: PyTree) -> PyTree:
    """Return a copy with all `Param`s set to `trainable=True`.

    >>> unfrozen_model = ion.tree.unfreeze(model)
    """

    def _unfreeze_leaf(leaf):
        if isinstance(leaf, Param) and not leaf.trainable:
            return Param(leaf.value, trainable=True)
        return leaf

    return jax.tree.map(_unfreeze_leaf, pytree, is_leaf=is_param)


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """Add optimizer deltas to a model's trainable parameters.

    >>> model = ion.tree.apply_updates(model, grads)
    """

    def _apply(param: Any, update: Any) -> Any:
        if update is None:
            return param
        if isinstance(param, Param) and not param.trainable:
            return param
        if isinstance(param, Param):
            delta = update.value if isinstance(update, Param) else update
            return Param(param.value + delta, trainable=param.trainable)
        return param + update

    return jax.tree.map(
        _apply,
        model,
        updates,
        is_leaf=lambda x: x is None or isinstance(x, Param),
    )


@jtu.register_pytree_node_class
class Static:
    """Wraps a value so JAX treats it as static metadata, not a traced array."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return [], self.value

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)


def save(path: str, pytree: PyTree) -> None:
    """Serialize a PyTree's array leaves to a `.npz` file.

    >>> ion.tree.save("model.npz", model)
    """
    flat_leaves, _ = jtu.tree_flatten(pytree)

    array_leaves = [
        np.asarray(leaf) for leaf in flat_leaves if isinstance(leaf, (jax.Array, np.ndarray))
    ]
    arrays_to_save = {str(i): arr for i, arr in enumerate(array_leaves)}

    np.savez(path, **arrays_to_save)  # type: ignore[call-overload]


def load(path: str, reference_pytree: PyTree) -> PyTree:
    """Load array leaves from a `.npz` file into a reference PyTree.

    >>> model = ion.tree.load("model.npz", model)
    """
    flat_leaves, tree_def = jtu.tree_flatten(reference_pytree)
    saved_data = np.load(path)

    loaded_leaves: list[Any] = []
    array_index = 0

    for leaf in flat_leaves:
        if isinstance(leaf, (jax.Array, np.ndarray)):
            loaded_leaves.append(jnp.array(saved_data[str(array_index)]))
            array_index += 1
        else:
            loaded_leaves.append(leaf)

    return tree_def.unflatten(loaded_leaves)
