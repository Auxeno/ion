"""Base class for neural network modules.

Classes:
    Module   Auto-registers subclasses as frozen JAX pytrees.

Subclassing `Module` gives you: dataclass conversion, pytree registration,
and immutability after `__init__`. No manual boilerplate needed.

See docs/internals.md for implementation details.
"""

import dataclasses
import functools
import zlib
from collections.abc import Iterable, Iterator
from typing import Any, Self

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from .. import tree
from ..tree import Static
from .param import Param


def _wrap_non_arrays(value: Any) -> Any:
    """Wrap non-array leaves in Static so JAX treats them as compile-time constants."""
    if isinstance(value, (Module, Param)):
        return value
    return jax.tree.map(
        lambda leaf: (
            leaf if isinstance(leaf, (Module, Param, jax.Array, np.ndarray)) else Static(leaf)
        ),
        value,
        is_leaf=lambda x: isinstance(x, (Module, Param, Static)),
    )


def _unwrap_non_arrays(value: Any) -> Any:
    """Inverse of `_wrap_non_arrays`. Strips Static wrappers back to plain values."""
    if isinstance(value, (Module, Param)):
        return value
    return jax.tree.map(
        lambda leaf: leaf.value if isinstance(leaf, Static) else leaf,
        value,
        is_leaf=lambda x: isinstance(x, (Module, Param, Static)),
    )


def _register_pytree(cls: type) -> None:
    """Register a class as a JAX pytree, wrapping non-array leaves as static metadata."""
    field_names = tuple(field.name for field in dataclasses.fields(cls))

    def flatten_with_keys(obj: Any) -> tuple[list[tuple[Any, Any]], None]:
        children = [
            (jtu.GetAttrKey(name), _wrap_non_arrays(getattr(obj, name))) for name in field_names
        ]
        return children, None

    def unflatten(_auxiliary_data: None, children: Iterable[Any]) -> Any:
        new_instance = object.__new__(cls)
        for name, value in zip(field_names, children):
            object.__setattr__(new_instance, name, _unwrap_non_arrays(value))
        object.__setattr__(new_instance, "_frozen", True)
        return new_instance

    jtu.register_pytree_with_keys(cls, flatten_with_keys, unflatten)


class Module:
    """Base class for neural network modules.

    Notes
    -----
    Subclasses are auto-converted to frozen dataclasses and registered as JAX pytrees.
    Instances are immutable after ``__init__``; use :meth:`replace` for modified copies.
    Non-array fields are wrapped as static pytree metadata automatically.

    Examples
    --------
    >>> model = Linear(3, 16, key=key)
    >>> new_model = model.replace(b=None)  # modified copy
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically convert subclasses into dataclasses and register them as PyTrees."""
        super().__init_subclass__(**kwargs)

        # Apply dataclass decorator, only generate __init__ if one doesn't exist
        has_custom_constructor = "__init__" in cls.__dict__
        dataclasses.dataclass(init=not has_custom_constructor, repr=False, eq=False)(cls)

        # Intercept the constructor to inject post-initialization freezing
        original_constructor = cls.__init__

        @functools.wraps(original_constructor)
        def _constructor_with_freeze(self: Any, *args: Any, **kwargs: Any) -> None:
            """Temporarily unfreeze, run subclass constructor and refreeze."""

            # A nesting depth counter ensures we only freeze at the outermost level
            depth = getattr(self, "_init_depth", 0)
            if depth == 0:
                object.__setattr__(self, "_frozen", False)
            object.__setattr__(self, "_init_depth", depth + 1)
            original_constructor(self, *args, **kwargs)
            object.__setattr__(self, "_init_depth", depth)
            if depth == 0:
                object.__setattr__(self, "_frozen", True)

        cls.__init__ = _constructor_with_freeze

        _register_pytree(cls)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment only during initialization."""
        # Default to False to allow assignment before _frozen is explicitly set
        if not getattr(self, "_frozen", False):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}', {type(self).__name__} is frozen after __init__."
            )

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion to maintain a consistent PyTree structure."""
        raise AttributeError(f"Cannot delete attribute '{name}', {type(self).__name__} is frozen.")

    def __iter__(self) -> Iterator[Any]:
        """Iterate over dataclass field values."""
        return (getattr(self, f.name) for f in dataclasses.fields(self))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Minimal textual pretty printing for pytrees."""

        fields = dataclasses.fields(self)  # type: ignore[reportArgumentType]
        if not fields:
            return f"{type(self).__name__}()"

        parts = [f"{type(self).__name__}("]
        for field in fields:
            value = getattr(self, field.name)
            if isinstance(value, Param):
                parts.append(f"  {field.name}={value!r},")
            elif hasattr(value, "shape") and hasattr(value, "dtype"):
                dtype = Param.short_dtype(value.dtype.name)
                parts.append(f"  {field.name}={dtype}{list(value.shape)},")
            elif (
                isinstance(value, (tuple, list))
                and value
                and any(isinstance(item, Module) for item in value)
            ):
                open_b, close_b = ("(", ")") if isinstance(value, tuple) else ("[", "]")
                parts.append(f"  {field.name}={open_b}")
                for item in value:
                    parts.append(f"    {repr(item).replace(chr(10), chr(10) + '    ')},")
                parts.append(f"  {close_b},")
            elif callable(value) and hasattr(value, "__name__"):
                parts.append(f"  {field.name}={value.__name__},")
            else:
                val_str = repr(value).replace("\n", "\n  ")
                parts.append(f"  {field.name}={val_str},")
        parts.append(")")
        return "\n".join(parts)

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to add color to Modules with Treescope."""
        import treescope

        child_attributes = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)  # type: ignore[reportArgumentType]
        }

        # Generate color from class name
        qualname = type(self).__qualname__
        salt = "5g157w"
        h = zlib.crc32(f"{salt}:{qualname}".encode())
        hue = (h % 10_000) / 10_000 * 360
        color = f"oklch(0.8 0.1 {hue:.1f})"

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=child_attributes,
            path=path,
            subtree_renderer=subtree_renderer,
            color=color,
        )

    def replace(self, **field_updates: Any) -> Self:
        """Return a new instance with the specified fields replaced.

        Parameters
        ----------
        **field_updates
            Field names and their new values. Must be existing dataclass fields.

        Returns
        -------
        Self
            Frozen copy with the specified fields replaced.

        Examples
        --------
        >>> new_model = model.replace(b=None)  # remove bias
        """

        # Ensure field exists in self
        valid_names = {field.name for field in dataclasses.fields(self)}  # type: ignore[reportArgumentType]
        unknown = field_updates.keys() - valid_names
        if unknown:
            raise ValueError(
                f"Unknown field(s) {unknown} for {type(self).__name__}. Valid fields: {valid_names}"
            )

        # Allocate a blank instance to avoid running initialization logic again
        new_instance = object.__new__(type(self))

        for field in dataclasses.fields(self):  # type: ignore[reportArgumentType]
            # Apply provided updates or fall back to the existing attribute values
            new_value = field_updates.get(field.name, getattr(self, field.name))
            object.__setattr__(new_instance, field.name, new_value)

        # Ensure the newly created copy is also immutable
        object.__setattr__(new_instance, "_frozen", True)
        return new_instance

    def freeze(self) -> Self:
        """Return a copy with all parameters frozen (non-trainable). Wraps `ion.freeze`.

        >>> frozen_model = model.freeze()
        """
        return tree.freeze(self)

    def unfreeze(self) -> Self:
        """Return a copy with all parameters unfrozen (trainable). Wraps `ion.unfreeze`.

        >>> unfrozen_model = model.unfreeze()
        """
        return tree.unfreeze(self)

    def astype(self, dtype: jax.numpy.dtype, *, params_only: bool = False) -> Self:
        """Return a copy with matching leaves cast to *dtype*. Wraps `ion.astype`.

        >>> bf16_model = model.astype(jnp.bfloat16)
        """
        return tree.astype(self, dtype, params_only=params_only)

    @property
    def params(self) -> PyTree:
        """Return only the `Param` leaves, replacing everything else with `None`.

        >>> model.params  # Param leaves only, rest is None
        """
        params = jtu.tree_map(
            lambda leaf: leaf if tree.is_param(leaf) else None,
            self,
            is_leaf=tree.is_param,
        )
        return params

    @property
    def num_params(self) -> int:
        """Total number of parameters (trainable and frozen).

        >>> model.num_params  # e.g. 101770
        """
        leaves = jtu.tree_leaves(self, is_leaf=tree.is_param)
        return sum(p._value.size for p in leaves if tree.is_param(p))
