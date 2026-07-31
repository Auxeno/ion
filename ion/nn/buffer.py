"""Non-trainable values updated during forward passes.

Classes:
    Buffers       Model buffers.
    BufferModule  Module with one buffer payload.
"""

import dataclasses
import functools
from typing import Any, Self

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree

from .module import Module
from .param import Param


@functools.partial(jtu.register_dataclass, data_fields=["_values"], meta_fields=["_keys"])
@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class Buffers:
    """Non-trainable model values updated during forward passes.

    Notes
    -----
    Buffers hold values such as BatchNorm running statistics. Create them with
    `Module.init_buffers` and pass the updated value through each forward call.

    A `BufferModule` reads its value with `buffers[self]` and updates it with
    `buffers.set(self, value)`.

    Examples
    --------
    >>> buffers = model.init_buffers()
    >>> y, buffers = model(x, buffers, training=True)
    >>> buffers  # Buffers((f32[64], f32[64]))
    """

    _keys: tuple[object, ...]
    _values: tuple[PyTree, ...]

    def __getitem__(self, module: "BufferModule") -> PyTree:
        try:
            value = self._values[self._keys.index(module._buffer_key)]
        except ValueError:
            raise ValueError(f"No {type(module).__name__} buffers; call init_buffers()") from None

        return jax.tree.map(jax.lax.stop_gradient, value)

    def set(self, module: "BufferModule", value: PyTree) -> "Buffers":
        old_value = self[module]
        layer_name = type(module).__name__

        # Updates must remain compatible with the initialized payload
        new_leaves = jax.tree.leaves(value)
        if not new_leaves or not all(
            isinstance(leaf, (jax.Array, np.ndarray)) for leaf in new_leaves
        ):
            raise TypeError(f"{layer_name} buffer must be a non-empty pytree of arrays")

        if jax.tree.structure(value) != jax.tree.structure(old_value):
            raise ValueError(f"{layer_name} buffer structure changed")

        old_leaves = jax.tree.leaves(old_value)
        for leaf_index, (old_leaf, new_leaf) in enumerate(zip(old_leaves, new_leaves)):
            if old_leaf.shape != new_leaf.shape:
                raise ValueError(
                    f"{layer_name} buffer {leaf_index} shape: {old_leaf.shape} -> {new_leaf.shape}"
                )
            if old_leaf.dtype != new_leaf.dtype:
                raise ValueError(
                    f"{layer_name} buffer {leaf_index} dtype: {old_leaf.dtype} -> {new_leaf.dtype}"
                )

        values = list(self._values)
        values[self._keys.index(module._buffer_key)] = jax.tree.map(jax.lax.stop_gradient, value)

        return Buffers(self._keys, tuple(values))

    def __repr__(self) -> str:
        """Minimal textual pretty printing for pytrees."""

        if not self._values:
            return "Buffers()"

        parts = ["Buffers("]
        for value in self._values:
            leaves = jax.tree.leaves(value)
            shapes = [f"{Param.short_dtype(leaf.dtype.name)}{list(leaf.shape)}" for leaf in leaves]
            joined = shapes[0] if len(shapes) == 1 else f"({', '.join(shapes)})"
            parts.append(f"  {joined},")
        parts.append(")")
        return "\n".join(parts)


class BufferModule(Module):
    """A module with one non-trainable buffer payload."""

    _buffer_key: object = dataclasses.field(init=False, repr=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls)
        object.__setattr__(instance, "_buffer_key", object())
        return instance

    def _init_buffer(self, *, key: PRNGKeyArray | None = None) -> PyTree:
        raise NotImplementedError(f"{type(self).__name__} must implement _init_buffer()")


def _init_buffers(model: Module, *, key: PRNGKeyArray | None = None) -> Buffers:
    is_buffer = lambda value: isinstance(value, BufferModule)
    discovered = [value for value in jax.tree.leaves(model, is_leaf=is_buffer) if is_buffer(value)]

    # Preserve traversal order while deduplicating shared buffer modules
    modules = list({module._buffer_key: module for module in discovered}.values())
    keys = [None] * len(modules) if key is None else jax.random.split(key, len(modules))

    values = []
    for module, layer_key in zip(modules, keys):
        layer_name = type(module).__name__
        fields = (
            getattr(module, field.name)
            for field in dataclasses.fields(module)  # type: ignore[reportArgumentType]
            if field.name != "_buffer_key"
        )
        nested = [
            value for value in jax.tree.leaves(tuple(fields), is_leaf=is_buffer) if is_buffer(value)
        ]
        if nested:
            raise ValueError(f"{layer_name} cannot contain another BufferModule")

        value = module._init_buffer(key=layer_key)
        leaves = jax.tree.leaves(value)
        if not leaves or not all(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves):
            raise TypeError(f"{layer_name} buffer must be a non-empty pytree of arrays")
        values.append(value)

    return Buffers(tuple(module._buffer_key for module in modules), tuple(values))
