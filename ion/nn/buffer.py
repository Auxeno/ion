"""Non-trainable values updated during forward passes.

Classes:
    Buffers       Model buffers.
    BufferModule  Module with one buffer value.
"""

import dataclasses
from collections.abc import Iterable
from typing import Any, Self

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree

from .module import Module
from .param import Param


@jtu.register_pytree_with_keys_class
@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class Buffers:
    """Non-trainable model values updated during forward passes.

    Notes
    -----
    Buffers hold values such as BatchNorm running statistics.
    Create buffers with `Module.init_buffers` and pass through each forward call.
    A `BufferModule` reads its value with `buffers[self]` and updates it with
    `buffers.set(self, value)`.

    Examples
    --------
    >>> buffers = model.init_buffers()
    >>> y, buffers = model(x, buffers, training=True)
    """

    _module_keys: tuple[object, ...]
    _path_info: tuple[tuple[str, str], ...]
    _values: tuple[PyTree, ...]

    def __getitem__(self, module: "BufferModule") -> PyTree:
        try:
            value = self._values[self._module_keys.index(module._buffer_key)]
        except ValueError:
            raise ValueError(
                f"No buffers found for {type(module).__name__}; call model.init_buffers()"
            ) from None

        return jax.tree.map(jax.lax.stop_gradient, value)

    def set(self, module: "BufferModule", value: PyTree) -> "Buffers":
        """Return a copy with `value` stored for `module`."""
        try:
            index = self._module_keys.index(module._buffer_key)
        except ValueError:
            raise ValueError(
                f"No buffers found for {type(module).__name__}; call model.init_buffers()"
            ) from None

        old_value = self._values[index]
        module_name = type(module).__name__
        path, label = self._path_info[index]
        location = "" if label == module_name else f" at '{path}'"

        # Updates must remain compatible with the initialized value
        new_leaves = jax.tree.leaves(value)
        if not new_leaves or not all(
            isinstance(leaf, (jax.Array, np.ndarray)) for leaf in new_leaves
        ):
            raise TypeError(f"{module_name} buffer{location} must be a non-empty pytree of arrays")

        if jax.tree.structure(value) != jax.tree.structure(old_value):
            raise ValueError(f"{module_name} buffer{location} structure changed")

        old_leaves = jax.tree.leaves(old_value)
        for leaf_index, (old_leaf, new_leaf) in enumerate(zip(old_leaves, new_leaves)):
            if old_leaf.shape != new_leaf.shape:
                raise ValueError(
                    f"{module_name} buffer{location} leaf {leaf_index} shape changed: "
                    f"{old_leaf.shape} -> {new_leaf.shape}"
                )
            if old_leaf.dtype != new_leaf.dtype:
                raise ValueError(
                    f"{module_name} buffer{location} leaf {leaf_index} dtype changed: "
                    f"{old_leaf.dtype} -> {new_leaf.dtype}"
                )

        values = list(self._values)
        values[index] = jax.tree.map(jax.lax.stop_gradient, value)

        return Buffers(self._module_keys, self._path_info, tuple(values))

    def tree_flatten_with_keys(self) -> tuple[list[tuple[Any, PyTree]], tuple]:
        children = [
            (jtu.GetAttrKey(path), value) for (path, _), value in zip(self._path_info, self._values)
        ]
        return children, (self._module_keys, self._path_info)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: Iterable[PyTree]) -> "Buffers":
        """Construct without initializing buffer values."""
        module_keys, path_info = aux
        return cls(module_keys, path_info, tuple(children))

    def __repr__(self) -> str:
        """Minimal textual pretty printing for pytrees."""
        if not self._values:
            return "Buffers()"

        parts = ["Buffers("]
        for (_, label), value in zip(self._path_info, self._values):
            leaves = jax.tree.leaves(value)
            shapes = [f"{Param.short_dtype(leaf.dtype.name)}{list(leaf.shape)}" for leaf in leaves]
            parts.append(f"  {label}({', '.join(shapes)}),")
        parts.append(")")
        return "\n".join(parts)

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to display buffer paths and values in grey with Treescope."""
        import treescope

        return treescope.repr_lib.render_dictionary_wrapper(
            object_type=type(self),
            wrapped_dict={label: value for (_, label), value in zip(self._path_info, self._values)},
            path=path,
            subtree_renderer=subtree_renderer,
            color="oklch(0.925 0.0 0.0)",
        )


class BufferModule(Module):
    """A module with one non-trainable buffer value."""

    _buffer_key: object = dataclasses.field(init=False, repr=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Preserve buffer identity across model transformations
        instance = super().__new__(cls)
        object.__setattr__(instance, "_buffer_key", object())
        return instance

    def _init_buffer(self, *, key: PRNGKeyArray | None = None) -> PyTree:
        """Initialize this module's buffer value. Subclasses should override this method."""
        raise NotImplementedError(f"{type(self).__name__} must implement _init_buffer()")


def _init_buffers(model: Module, *, key: PRNGKeyArray | None = None) -> Buffers:
    """Initialize buffer values in a model."""
    is_buffer = lambda value: isinstance(value, BufferModule)

    # Preserve traversal order while deduplicating shared buffer modules
    buffer_modules: dict[object, tuple[list[str], BufferModule]] = {}
    for tree_keys, module in jtu.tree_leaves_with_path(model, is_leaf=is_buffer):
        if is_buffer(module):
            paths, _ = buffer_modules.setdefault(module._buffer_key, ([], module))
            paths.append(jtu.keystr(tree_keys).lstrip("."))

    # Split one initialization key per unique buffer module
    init_keys = (
        [None] * len(buffer_modules) if key is None else jax.random.split(key, len(buffer_modules))
    )

    # Initialize and validate each buffer value
    module_keys, path_info, buffer_values = [], [], []
    for (paths, module), init_key in zip(buffer_modules.values(), init_keys):
        module_name = type(module).__name__
        visible_paths = " | ".join(path for path in paths if path)
        label = f"{visible_paths}: {module_name}" if visible_paths else module_name
        location = f" at '{visible_paths}'" if visible_paths else ""
        if any(is_buffer(value) for value in jax.tree.leaves(tuple(module), is_leaf=is_buffer)):
            raise ValueError(f"{module_name}{location} cannot contain another BufferModule")

        value = module._init_buffer(key=init_key)
        leaves = jax.tree.leaves(value)
        if not leaves or not all(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves):
            raise TypeError(f"{module_name} buffer{location} must be a non-empty pytree of arrays")
        module_keys.append(module._buffer_key)
        path_info.append((paths[0] or module_name, label))
        buffer_values.append(value)

    return Buffers(tuple(module_keys), tuple(path_info), tuple(buffer_values))
