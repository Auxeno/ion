"""Param-aware optimizer wrapping an optax GradientTransformation.

Classes:
    Optimizer   Wraps optax with auto-partitioning for frozen Params.

See docs/internals.md for implementation details.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import PyTree

from .nn.param import Param
from .tree import is_param


def _apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """Add optimizer deltas to trainable `Param` leaves in a model pytree."""

    def _apply(param: Any, update: Any) -> Any:
        if update is None:
            return param
        if not isinstance(param, Param):
            return param
        if not param.trainable:
            return param
        # Unwrap Param updates to get the raw delta array
        delta = update._value if isinstance(update, Param) else update
        return Param(param._value + delta, trainable=param.trainable)

    return jax.tree.map(
        _apply,
        model,
        updates,
        is_leaf=lambda x: x is None or isinstance(x, Param),
    )


def _auto_partition(
    transform: optax.GradientTransformation,
    model: PyTree,
) -> optax.GradientTransformation:
    """Wrap transform with `optax.partition` if model has non-trainable array leaves."""

    # Skip allocating optimizer state for frozen and non-Param leaves to save memory
    leaves = jax.tree.leaves(model, is_leaf=is_param)
    if all(isinstance(leaf, Param) and leaf.trainable for leaf in leaves):
        return transform

    def _label(leaf: Any) -> Any:
        if isinstance(leaf, Param):
            return Param(
                "train" if leaf.trainable else "freeze",  # type: ignore[arg-type]
                trainable=leaf.trainable,
            )
        return "freeze"

    return optax.partition(
        transforms={"train": transform, "freeze": optax.set_to_zero()},
        param_labels=lambda params: jax.tree.map(
            _label,
            params,
            is_leaf=is_param,
        ),
    )


@jtu.register_pytree_node_class
class Optimizer:
    """Wraps an optax optimizer with Param-aware updates.

    Parameters
    ----------
    transform : optax.GradientTransformation
        An optax optimizer (e.g. `optax.adam(3e-4)`).
    model : PyTree
        Model to optimize. Frozen and non-Param leaves are auto-partitioned out.

    Examples
    --------
    >>> optimizer = ion.Optimizer(optax.adam(3e-4), model)
    >>> model, optimizer = optimizer.update(model, grads)
    """

    __slots__ = ("_transform", "state", "step")

    def __init__(self, transform: optax.GradientTransformation, model: PyTree) -> None:
        self._transform = _auto_partition(transform, model)
        self.state = self._transform.init(model)
        self.step = jnp.array(0, dtype=jnp.int32)

    def update(self, model: PyTree, grads: PyTree, **kwargs) -> tuple[PyTree, "Optimizer"]:
        """Apply gradients to the model and advance optimizer state.

        Parameters
        ----------
        model : PyTree
            Current model.
        grads : PyTree
            Gradients from `jax.grad`.
        **kwargs
            Extra arguments forwarded to the optax transform's `update`.

        Returns
        -------
        tuple[PyTree, Optimizer]
            Updated model and optimizer.
        """
        updates, new_state = self._transform.update(
            grads,
            self.state,
            model,
            **kwargs,
        )
        new_model = _apply_updates(model, updates)
        return new_model, Optimizer._new(
            self._transform,
            new_state,
            self.step + 1,
        )

    @classmethod
    def _new(
        cls,
        transform: optax.GradientTransformation,
        state: optax.OptState,
        step: jax.Array,
    ) -> "Optimizer":
        """Construct without running `__init__` (for unflatten and update)."""
        obj = object.__new__(cls)
        obj._transform = transform
        obj.state = state
        obj.step = step
        return obj

    def tree_flatten(self) -> tuple[tuple, optax.GradientTransformation]:
        return (self.state, self.step), self._transform

    @classmethod
    def tree_unflatten(cls, aux: optax.GradientTransformation, children: tuple) -> "Optimizer":
        state, step = children
        return cls._new(aux, state, step)

    def __repr__(self) -> str:
        """Minimal textual pretty printing."""
        step_val = self.step.item() if hasattr(self.step, "item") else self.step
        num_leaves = len(jax.tree.leaves(self.state))
        return f"Optimizer(step={step_val}, state_leaves={num_leaves})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to add color to Optimizers with Treescope."""
        import treescope

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes={"step": self.step, "state": self.state},
            path=path,
            subtree_renderer=subtree_renderer,
            color="oklch(0.88 0.10 95)",
        )
