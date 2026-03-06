"""PyTree-aware wrappers around `jax.grad` and `jax.value_and_grad`.

Functions:
    grad            Differentiate only trainable Param leaves.
    value_and_grad  Like grad, but also returns the function output.

These wrappers handle the split-transform-recombine cycle so only trainable
`Param` leaves are differentiated, while everything else is held constant.

See docs/internals.md for implementation details.
"""

import functools
from collections.abc import Callable
from typing import Any, Sequence

import jax
import jax.tree_util as jtu

from .tree import is_param, is_trainable_param


def _split_leaves(
    flat_leaves: list[Any],
    predicate: Callable[[Any], bool],
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[bool, ...]]:
    """Partition flat leaves into matches, non-matches, and a positional mask."""

    # Records each leaf's original position so _merge_leaves can restore order
    matching_leaves: list[Any] = []
    non_matching_leaves: list[Any] = []
    is_match_mask: list[bool] = []

    for leaf in flat_leaves:
        is_match = predicate(leaf)
        is_match_mask.append(is_match)
        if is_match:
            matching_leaves.append(leaf)
        else:
            non_matching_leaves.append(leaf)

    return tuple(matching_leaves), tuple(non_matching_leaves), tuple(is_match_mask)


def _merge_leaves(
    matching_leaves: tuple[Any, ...],
    non_matching_leaves: tuple[Any, ...],
    is_match_mask: tuple[bool, ...],
) -> list[Any]:
    """Inverse of `_split_leaves`. Reconstructs the original leaf order."""
    reconstructed_leaves: list[Any] = []

    # Walk the mask to pick from the correct group at each position
    match_index, non_match_index = 0, 0

    for is_match in is_match_mask:
        if is_match:
            reconstructed_leaves.append(matching_leaves[match_index])
            match_index += 1
        else:
            reconstructed_leaves.append(non_matching_leaves[non_match_index])
            non_match_index += 1

    return reconstructed_leaves


def grad(
    fn: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable[..., Any]:
    """Like `jax.grad`, but differentiates only trainable `Param` leaves.

    >>> grads = ion.grad(loss_fn)(model, x, y)
    >>> grads_a, grads_b = ion.grad(loss_fn, argnums=(0, 1))(model_a, model_b, x)
    >>> grads, aux = ion.grad(loss_fn, has_aux=True)(model, x, y)
    """
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Split each target into (trainable, static, mask, tree_def)
        splits = []
        for i in argnums_tup:
            flat, tree_def = jtu.tree_flatten(args[i], is_leaf=is_param)
            trainable, static, mask = _split_leaves(flat, is_trainable_param)
            splits.append((trainable, static, mask, tree_def))

        all_trainable = tuple(s[0] for s in splits)

        def inner(diff: tuple[tuple[Any, ...], ...]) -> Any:
            rebuilt = list(args)
            for i, (_, static, mask, tree_def) in zip(argnums_tup, splits):
                rebuilt[i] = tree_def.unflatten(
                    _merge_leaves(diff[argnums_tup.index(i)], static, mask)
                )
            return fn(*rebuilt, **kwargs)

        if has_aux:
            grads_raw, aux = jax.grad(inner, has_aux=True, holomorphic=holomorphic)(all_trainable)
        else:
            grads_raw = jax.grad(inner, holomorphic=holomorphic)(all_trainable)
            aux = None

        grad_trees = tuple(
            td.unflatten(_merge_leaves(g, tuple(None for _ in s), m))
            for g, (_, s, m, td) in zip(grads_raw, splits)
        )

        result = grad_trees if not isinstance(argnums, int) else grad_trees[0]
        return (result, aux) if has_aux else result

    return wrapper


def value_and_grad(
    fn: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable[..., Any]:
    """Like `jax.value_and_grad`, but differentiates only trainable `Param` leaves.

    >>> loss, grads = ion.value_and_grad(loss_fn)(model, x, y)
    >>> loss, (grads_a, grads_b) = ion.value_and_grad(loss_fn, argnums=(0, 1))(a, b, x)
    >>> (loss, aux), grads = ion.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    """
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        splits = []
        for i in argnums_tup:
            flat, tree_def = jtu.tree_flatten(args[i], is_leaf=is_param)
            trainable, static, mask = _split_leaves(flat, is_trainable_param)
            splits.append((trainable, static, mask, tree_def))

        all_trainable = tuple(s[0] for s in splits)

        def inner(diff: tuple[tuple[Any, ...], ...]) -> Any:
            rebuilt = list(args)
            for i, (_, static, mask, tree_def) in zip(argnums_tup, splits):
                rebuilt[i] = tree_def.unflatten(
                    _merge_leaves(diff[argnums_tup.index(i)], static, mask)
                )
            return fn(*rebuilt, **kwargs)

        value, grads_raw = jax.value_and_grad(inner, has_aux=has_aux, holomorphic=holomorphic)(
            all_trainable
        )

        grad_trees = tuple(
            td.unflatten(_merge_leaves(g, tuple(None for _ in s), m))
            for g, (_, s, m, td) in zip(grads_raw, splits)
        )

        result = grad_trees if not isinstance(argnums, int) else grad_trees[0]
        return value, result

    return wrapper
