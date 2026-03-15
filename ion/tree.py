"""Predicate-based utilities for filtering JAX pytrees.

Functions:
    is_param            Check if a leaf is a Param.
    is_trainable_param  Check if a leaf is a trainable Param.
    apply_updates       Add optimizer deltas to trainable parameters.
    astype              Cast all floating-point leaves to a given dtype.
    freeze              Set all Params to trainable=False.
    unfreeze            Set all Params to trainable=True.

Classes:
    _Static             Wraps a value so JAX treats it as static metadata (internal).

Neural network pytrees mix `Param` wrappers, plain arrays, and static Python
values. Standard `jax.tree_util` treats every leaf uniformly, these utilities
provide selective filtering by type and trainability.

See docs/internals.md for implementation details.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from .nn.param import Param


def is_param(x: Any) -> bool:
    """Check if an object is a `Param`."""
    return isinstance(x, Param)


def is_trainable_param(x: Any) -> bool:
    """Check if an object is a trainable `Param`."""
    return isinstance(x, Param) and x.trainable


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """Add optimizer deltas to a model's trainable parameters.

    Parameters
    ----------
    model : PyTree
        Model pytree containing `Param` leaves.
    updates : PyTree
        Matching-structure pytree of gradient or optimizer deltas.

    Returns
    -------
    PyTree
        Updated model with `Param` wrappers preserved.

    Notes
    -----
    Frozen params, non-`Param` leaves, and `None` updates are skipped.

    Examples
    --------
    >>> model = ion.tree.apply_updates(model, grads)
    """

    def _apply(param: Any, update: Any) -> Any:
        if update is None:
            return param
        if not isinstance(param, Param):
            return param
        if not param.trainable:
            return param
        delta = update._value if isinstance(update, Param) else update
        return Param(param._value + delta, trainable=param.trainable)

    return jax.tree.map(
        _apply,
        model,
        updates,
        is_leaf=lambda x: x is None or isinstance(x, Param),
    )


def astype(pytree: PyTree, dtype: jnp.dtype, *, params_only: bool = False) -> PyTree:
    """Cast all leaves in a pytree whose dtype matches the target's family.

    If a float dtype is given, only float leaves are cast; likewise for
    complex and integer dtypes. Other families are left unchanged.

    Parameters
    ----------
    pytree : PyTree
        Pytree containing `Param` wrappers, plain arrays, or both.
    dtype : jnp.dtype
        Target dtype; its family controls which leaves are cast.
    params_only : bool, optional
        If `True`, only `Param` leaves are cast. Default `False`.

    Returns
    -------
    PyTree
        Pytree with matching leaves cast and `Param` wrappers preserved.

    Examples
    --------
    >>> bf16_model = ion.astype(model, jnp.bfloat16)
    >>> bf16_params = ion.astype(model, jnp.bfloat16, params_only=True)
    """

    # Only apply to leaves of same family as provided dtype
    family = (
        jnp.complexfloating
        if jnp.issubdtype(dtype, jnp.complexfloating)
        else jnp.integer
        if jnp.issubdtype(dtype, jnp.integer)
        else jnp.floating
    )

    def _cast(leaf: Any) -> Any:
        if isinstance(leaf, Param):
            if jnp.issubdtype(leaf._value.dtype, family):
                return Param(leaf._value.astype(dtype), trainable=leaf.trainable)
            return leaf
        if not params_only and isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, family):
            return leaf.astype(dtype)
        return leaf

    return jax.tree.map(_cast, pytree, is_leaf=is_param)


def freeze(pytree: PyTree) -> PyTree:
    """Return a copy with all `Param`s set to `trainable=False`.

    >>> frozen_model = ion.tree.freeze(model)
    """

    def _freeze_leaf(leaf):
        if isinstance(leaf, Param) and leaf.trainable:
            return Param(leaf._value, trainable=False)
        return leaf

    return jax.tree.map(_freeze_leaf, pytree, is_leaf=is_param)


def unfreeze(pytree: PyTree) -> PyTree:
    """Return a copy with all `Param`s set to `trainable=True`.

    >>> unfrozen_model = ion.tree.unfreeze(model)
    """

    def _unfreeze_leaf(leaf):
        if isinstance(leaf, Param) and not leaf.trainable:
            return Param(leaf._value, trainable=True)
        return leaf

    return jax.tree.map(_unfreeze_leaf, pytree, is_leaf=is_param)


@jtu.register_pytree_node_class
class _Static:
    """Wraps a value so JAX treats it as static metadata, not a traced array."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return [], self.value

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)
