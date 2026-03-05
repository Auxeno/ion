"""PyTree-aware wrappers around `jax.grad` and `jax.value_and_grad`.

Functions:
    grad            Differentiate only trainable Param leaves.
    value_and_grad  Like grad, but also returns the function output.

These wrappers handle the split-transform-recombine cycle so only trainable
`Param` leaves are differentiated, while everything else is held constant.

See docs/internals.md for implementation details.
"""

import functools
from typing import Any, Callable

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
    """Inverse of `_split_leaves` — reconstruct the original leaf order."""
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


def grad(fn: Callable[..., Any], *, has_aux: bool = False) -> Callable[..., Any]:
    """Like `jax.grad`, but differentiates only trainable `Param` leaves.

    >>> grads = ion.grad(loss_fn)(model, x, y)
    >>> grads, aux = ion.grad(loss_fn, has_aux=True)(model, x, y)
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        target_pytree, *remaining_args = args

        flat_leaves, tree_def = jtu.tree_flatten(target_pytree, is_leaf=is_param)
        trainable_params, static_leaves, is_match_mask = _split_leaves(
            flat_leaves, is_trainable_param
        )

        def inner(diff_leaves: tuple[Any, ...]) -> Any:
            """Rebuild the original object and execute the function."""
            all_leaves = _merge_leaves(diff_leaves, static_leaves, is_match_mask)
            reconstructed_pytree = tree_def.unflatten(all_leaves)
            return fn(reconstructed_pytree, *remaining_args, **kwargs)

        empty_padding = tuple(None for _ in static_leaves)

        if has_aux:
            computed_gradients, aux = jax.grad(inner, has_aux=True)(trainable_params)
            flat_gradient_leaves = _merge_leaves(computed_gradients, empty_padding, is_match_mask)
            return tree_def.unflatten(flat_gradient_leaves), aux

        computed_gradients = jax.grad(inner)(trainable_params)
        flat_gradient_leaves = _merge_leaves(computed_gradients, empty_padding, is_match_mask)
        return tree_def.unflatten(flat_gradient_leaves)

    return wrapper


def value_and_grad(fn: Callable[..., Any], *, has_aux: bool = False) -> Callable[..., Any]:
    """Like `jax.value_and_grad`, but differentiates only trainable `Param` leaves.

    >>> loss, grads = ion.value_and_grad(loss_fn)(model, x, y)
    >>> (loss, aux), grads = ion.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        target_pytree, *remaining_args = args

        flat_leaves, tree_def = jtu.tree_flatten(target_pytree, is_leaf=is_param)
        trainable_params, static_leaves, is_match_mask = _split_leaves(
            flat_leaves, is_trainable_param
        )

        def inner(diff_leaves: tuple[Any, ...]) -> Any:
            all_leaves = _merge_leaves(diff_leaves, static_leaves, is_match_mask)
            reconstructed_pytree = tree_def.unflatten(all_leaves)
            return fn(reconstructed_pytree, *remaining_args, **kwargs)

        value, computed_gradients = jax.value_and_grad(inner, has_aux=has_aux)(trainable_params)

        empty_padding = tuple(None for _ in static_leaves)
        flat_gradient_leaves = _merge_leaves(computed_gradients, empty_padding, is_match_mask)

        return value, tree_def.unflatten(flat_gradient_leaves)

    return wrapper
