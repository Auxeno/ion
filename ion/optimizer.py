"""Param-aware optimizer wrapping an optax GradientTransformation.

Classes:
    Optimizer   Wraps optax with auto-partitioning for frozen Params.

Supports per-field transforms via a dict mapping field names to transforms.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from .nn.buffer import Buffer
from .nn.param import Param
from .tree import is_buffer, is_param
from .typing import PyTree


def _without_buffers(pytree: PyTree) -> PyTree:
    """Replace `Buffer` fields with `None` before passing a pytree to optax."""
    return jax.tree.map(
        lambda leaf: None if isinstance(leaf, Buffer) else leaf,
        pytree,
        is_leaf=is_buffer,
    )


def _apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """Add optimizer deltas to trainable `Param` leaves in a model pytree."""

    def _apply(param: Any, update: Any) -> Any:
        if not isinstance(param, Param) or not param.trainable or update is None:
            return param
        delta = update._value if isinstance(update, Param) else update
        return Param(param._value + delta, trainable=True)

    return jax.tree.map(
        _apply,
        model,
        updates,
        is_leaf=lambda x: x is None or isinstance(x, (Param, Buffer)),
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
                f"Field '{field_name}' has no transform, transforms cover {sorted(field_to_label)}"
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
        Optax optimizer or dict mapping field names to per-field transforms.
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

    __slots__ = ("_transform", "_fields", "_structure", "state", "step")

    def __init__(
        self,
        tx: optax.GradientTransformation
        | dict[str | tuple[str, ...], optax.GradientTransformation],
        model: PyTree,
    ) -> None:
        model_without_buffers = _without_buffers(model)
        if isinstance(tx, dict):
            self._transform, self._fields = _field_partition(tx)
        else:
            self._transform = _auto_partition(tx, model_without_buffers)
            self._fields = None
        self._structure = jax.tree.structure(model_without_buffers)
        self.state = self._transform.init(model_without_buffers)
        self.step = jnp.array(0, dtype=jnp.uint32)

    def update(self, model: PyTree, grads: PyTree, **kwargs: Any) -> tuple[PyTree, "Optimizer"]:
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
        model_without_buffers = _without_buffers(model)
        if jax.tree.structure(model_without_buffers) != self._structure:
            raise ValueError("Model structure or trainability changed, create a new Optimizer")

        grads_without_buffers = _without_buffers(grads)
        updates, new_state = self._transform.update(
            grads_without_buffers, self.state, model_without_buffers, **kwargs
        )
        new_model = _apply_updates(model, updates)
        aux = (self._transform, self._fields, self._structure)
        return new_model, Optimizer.tree_unflatten(aux, (new_state, self.step + 1))

    def tree_flatten(self) -> tuple[tuple, tuple]:
        return (self.state, self.step), (self._transform, self._fields, self._structure)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: tuple) -> "Optimizer":
        """Construct without running `__init__` (also used by `update`)."""
        obj = object.__new__(cls)
        obj._transform, obj._fields, obj._structure = aux
        obj.state, obj.step = children
        return obj

    def __repr__(self) -> str:
        """Hook to summarize the state and render for the terminal."""
        from . import display

        return display.optimizer_repr(self)

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to summarize Optimizers and add color with Treescope."""
        from . import display

        return display.optimizer_treescope(self, path, subtree_renderer)
