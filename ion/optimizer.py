"""Param-aware optimizer wrapping an optax GradientTransformation.

Classes:
    Optimizer   Wraps optax with auto-partitioning for frozen Params.

Supports per-field transforms via a dict mapping field names to transforms.

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
        # Skip update
        if not isinstance(param, Param) or not param.trainable or update is None:
            return param

        # Apply update and rewrap as Param
        delta = update._value if isinstance(update, Param) else update
        return Param(param._value + delta, trainable=True)

    return jax.tree.map(
        _apply,
        model,
        updates,
        is_leaf=lambda x: x is None or isinstance(x, Param),
    )


def _auto_partition(
    tx: optax.GradientTransformation,
    model: PyTree,
) -> optax.GradientTransformation:
    """Wrap tx with `optax.partition` if model has non-trainable array leaves."""

    # Skip allocating optimizer state for frozen and non-Param leaves to save memory
    leaves = jax.tree.leaves(model, is_leaf=is_param)
    if all(isinstance(leaf, Param) and leaf.trainable for leaf in leaves):
        return tx

    def _label(leaf: Any) -> Any:
        if isinstance(leaf, Param):
            return Param(
                "train" if leaf.trainable else "freeze",  # type: ignore[arg-type]
                trainable=leaf.trainable,
            )
        return "freeze"

    return optax.partition(
        transforms={"train": tx, "freeze": optax.set_to_zero()},
        param_labels=lambda params: jax.tree.map(
            _label,
            params,
            is_leaf=is_param,
        ),
    )


def _field_partition(
    transforms: dict[str | tuple[str, ...], optax.GradientTransformation],
) -> tuple[optax.GradientTransformation, tuple[str, ...]]:
    """Route different optax transforms to top-level model fields."""
    field_to_label: dict[str, str] = {}
    groups: dict[str, optax.GradientTransformation] = {"__frozen__": optax.set_to_zero()}

    # Map each field to a group label
    for key, tx in transforms.items():
        label = str(key)
        groups[label] = tx
        fields = (key,) if isinstance(key, str) else key
        for field in fields:
            if field in field_to_label:
                raise ValueError(f"Field '{field}' appears in multiple transform groups")
            field_to_label[field] = label

    # Label each leaf by its top-level field name
    def _label(path: tuple, leaf: Any) -> Any:
        if not isinstance(leaf, Param):
            return "__frozen__"
        if not leaf.trainable:
            return Param("__frozen__", trainable=False)  # type: ignore[arg-type]
        field_name = path[0].name
        if field_name not in field_to_label:
            raise ValueError(
                f"Field '{field_name}' has trainable params but no transform. "
                f"Covered fields: {sorted(field_to_label)}"
            )
        return Param(field_to_label[field_name], trainable=True)  # type: ignore[arg-type]

    # Construct the partitioned transform
    tx = optax.partition(
        transforms=groups,  # type: ignore[arg-type]
        param_labels=lambda p: jax.tree.map_with_path(_label, p, is_leaf=is_param),
    )
    return tx, tuple(field_to_label)


@jtu.register_pytree_node_class
class Optimizer:
    """Wraps an optax optimizer with Param-aware updates.

    Parameters
    ----------
    tx : optax.GradientTransformation or dict
        A single optax optimizer (e.g. ``optax.adam(3e-4)``), or a dict mapping
        field names (or tuples of field names) to per-field transforms.
    model : PyTree
        Model to optimize. Frozen and non-Param leaves are auto-partitioned out.

    Examples
    --------
    >>> optimizer = ion.Optimizer(optax.adam(3e-4), model)
    >>> model, optimizer = optimizer.update(model, grads)

    Per-field transforms (e.g. different LRs for a GAN):

    >>> optimizer = ion.Optimizer(
    ...     {"generator": optax.adam(1e-4), "discriminator": optax.adam(4e-4)},
    ...     model,
    ... )
    """

    __slots__ = ("_transform", "_fields", "state", "step")

    def __init__(
        self,
        tx: optax.GradientTransformation
        | dict[str | tuple[str, ...], optax.GradientTransformation],
        model: PyTree,
    ) -> None:
        if isinstance(tx, dict):
            self._transform, self._fields = _field_partition(tx)
        else:
            self._transform = _auto_partition(tx, model)
            self._fields = None
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
            self._fields,
            new_state,
            self.step + 1,
        )

    @classmethod
    def _new(
        cls,
        tx: optax.GradientTransformation,
        fields: tuple[str, ...] | None,
        state: optax.OptState,
        step: jax.Array,
    ) -> "Optimizer":
        """Construct without running `__init__` (for unflatten and update)."""
        obj = object.__new__(cls)
        obj._transform = tx
        obj._fields = fields
        obj.state = state
        obj.step = step
        return obj

    def tree_flatten(self) -> tuple[tuple, tuple]:
        return (self.state, self.step), (self._transform, self._fields)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: tuple) -> "Optimizer":
        tx, fields = aux
        state, step = children
        return cls._new(tx, fields, state, step)

    def __repr__(self) -> str:
        """Minimal textual pretty printing."""
        step_val = self.step.item() if hasattr(self.step, "item") else self.step
        num_leaves = len(jax.tree.leaves(self.state))
        fields_str = f", fields={list(self._fields)}" if self._fields is not None else ""
        return f"Optimizer(step={step_val}, state_leaves={num_leaves}{fields_str})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to add color to Optimizers with Treescope."""
        import treescope

        attributes: dict[str, Any] = {"step": self.step, "state": self.state}
        if self._fields is not None:
            attributes["fields"] = self._fields

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=attributes,
            path=path,
            subtree_renderer=subtree_renderer,
            color="oklch(0.88 0.10 95)",
        )
