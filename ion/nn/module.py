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
from typing import Any, Generic, Self, TypeVar

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from .. import tree
from .param import Param

_M = TypeVar("_M")


@jtu.register_pytree_node_class
class _Static:
    """Wraps a value so JAX treats it as static metadata, not a traced array.

    Used internally by _register_module_as_pytree for non-array elements inside mixed containers
    (e.g. a callable sitting alongside Modules in a tuple).
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return [], self.value

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)


class _At(Generic[_M]):
    """Records an attribute/index path into a model and rebuilds it on `set`.

    Returned by `Module.at`. Each attribute or index access appends one step to the
    path; `set` reconstructs the modules and containers along it and shares every
    untouched subtree with the original.
    """

    __slots__ = ("_target", "_path")

    def __init__(self, target: _M, path: tuple = ()) -> None:
        self._target = target
        self._path = path

    def __getattr__(self, name: str) -> "_At[_M]":
        return _At(self._target, (*self._path, name))

    def __getitem__(self, key: Any) -> "_At[_M]":
        return _At(self._target, (*self._path, key))

    def set(self, value: Any) -> _M:
        """Return a copy of the target with `value` stored at this path."""

        def rebuild(node: Any, path: tuple) -> Any:
            if not path:
                return value
            step, rest = path[0], path[1:]
            if isinstance(node, Module):
                fields = dataclasses.fields(node)  # type: ignore[reportArgumentType]
                if step not in (field.name for field in fields):
                    raise AttributeError(f"{type(node).__name__} has no field '{step}'")
                # Copy fields onto a blank instance to avoid re-running __init__
                new_node = object.__new__(type(node))
                for field in fields:
                    object.__setattr__(new_node, field.name, getattr(node, field.name))
                object.__setattr__(new_node, step, rebuild(getattr(node, step), rest))
                object.__setattr__(new_node, "_frozen", True)
                return new_node
            if isinstance(node, (tuple, list)):
                # Namedtuples rebuild by position and accept field-name steps
                names = getattr(node, "_fields", None)
                index = names.index(step) if names and isinstance(step, str) else step
                items = list(node)
                items[index] = rebuild(items[index], rest)
                return type(node)(*items) if names else type(node)(items)
            if isinstance(node, dict):
                return {**node, step: rebuild(node[step], rest)}
            raise TypeError(f"Cannot set '{step}' inside {type(node).__name__}")

        return rebuild(self._target, self._path)


def _register_module_as_pytree(cls: type) -> Any:
    """Register a Module subclass as a JAX pytree."""
    array_like = (jax.Array, np.ndarray, Param, Module)
    field_names = tuple(field.name for field in dataclasses.fields(cls))

    def _classify(obj: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        """Classify each field as a dynamic child or static aux. Returns cached info."""
        child_info: list[tuple[str, str]] = []
        static_names: list[str] = []

        for name in field_names:
            # getattr falls back to class-level defaults for fields not assigned in __init__
            try:
                value = getattr(obj, name)
            except AttributeError:
                raise AttributeError(
                    f"Field '{name}' of {type(obj).__name__} was never assigned and has no default"
                ) from None

            if isinstance(value, array_like):
                child_info.append((name, "leaf"))
            elif isinstance(value, (tuple, list, dict)):
                leaves = jax.tree.leaves(value, is_leaf=lambda x: isinstance(x, array_like))
                arrays = sum(isinstance(x, array_like) for x in leaves)
                if arrays:
                    # Pure containers traverse natively, mixed ones need _Static wrapping
                    child_info.append((name, "container" if arrays == len(leaves) else "mixed"))
                else:
                    static_names.append(name)
            else:
                static_names.append(name)

        return tuple(child_info), tuple(static_names)

    def flatten_with_keys(obj: Any) -> tuple[list[tuple[Any, Any]], tuple]:
        child_info = obj.__dict__.get("_child_info")
        if child_info is None:
            child_info, static_names = _classify(obj)
        else:
            static_names = obj._static_names

        children = []
        for name, kind in child_info:
            value = getattr(obj, name)
            if kind == "mixed":
                value = jax.tree.map(
                    lambda x: x if isinstance(x, array_like) else _Static(x),
                    value,
                    is_leaf=lambda x: isinstance(x, array_like),
                )
            children.append((jtu.GetAttrKey(name), value))

        static_values = tuple(getattr(obj, name) for name in static_names)
        return children, (child_info, static_names, static_values)

    def unflatten(aux: tuple, children: Iterable[Any]) -> Any:
        child_info, static_names, static_values = aux
        new_instance = object.__new__(cls)

        # Restore dynamic children, unwrapping _Static in mixed containers
        for (name, kind), value in zip(child_info, children):
            if kind == "mixed":
                value = jax.tree.map(
                    lambda x: x.value if isinstance(x, _Static) else x,
                    value,
                    is_leaf=lambda x: isinstance(x, (_Static, *array_like)),
                )
            object.__setattr__(new_instance, name, value)

        # Restore static fields directly
        for name, value in zip(static_names, static_values):
            object.__setattr__(new_instance, name, value)

        object.__setattr__(new_instance, "_child_info", child_info)
        object.__setattr__(new_instance, "_static_names", static_names)
        object.__setattr__(new_instance, "_frozen", True)
        return new_instance

    jtu.register_pytree_with_keys(cls, flatten_with_keys, unflatten)
    return _classify


class Module:
    """Base class for neural network modules.

    Notes
    -----
    Subclasses are auto-converted to frozen dataclasses and registered as JAX pytrees.
    Instances are immutable after `__init__`; use `at` for modified copies.
    Non-array fields are wrapped as static pytree metadata automatically.

    Examples
    --------
    >>> model = Linear(3, 16, key=key)
    >>> new_model = model.at.b.set(None)  # modified copy
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass. Subclasses should override this method."""
        raise NotImplementedError(f"{type(self).__name__} must implement __call__")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically convert subclasses into dataclasses and register them as PyTrees."""
        super().__init_subclass__(**kwargs)

        # Apply dataclass decorator, only generate __init__ if one doesn't exist
        has_custom_constructor = "__init__" in cls.__dict__
        dataclasses.dataclass(init=not has_custom_constructor, repr=False, eq=False)(cls)

        # Register as pytree first so _classify is available to the constructor
        _classify = _register_module_as_pytree(cls)

        # Intercept the constructor to inject post-initialization freezing
        original_constructor = cls.__init__
        allowed = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]

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
                # Unannotated attributes would silently vanish on the first unflatten
                undeclared = vars(self).keys() - allowed - {"_frozen", "_init_depth"}
                if undeclared:
                    raise AttributeError(f"Unannotated field(s) {sorted(undeclared)} in __init__")
                child_info, static_names = _classify(self)
                object.__setattr__(self, "_child_info", child_info)
                object.__setattr__(self, "_static_names", static_names)
                object.__setattr__(self, "_frozen", True)

        cls.__init__ = _constructor_with_freeze

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute assignment only during initialization."""
        # Default to False to allow assignment before _frozen is explicitly set
        if not getattr(self, "_frozen", False):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}', {type(self).__name__} is frozen after __init__"
            )

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion to maintain a consistent PyTree structure."""
        raise AttributeError(f"Cannot delete attribute '{name}', {type(self).__name__} is frozen")

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

    @property
    def at(self) -> _At[Self]:
        """Path-based model surgery. Navigate to any field, index, or key, then `set`.

        Rebuilds only the nodes along the path; untouched subtrees are shared with
        the original. Works uniformly for arrays, `Param`s, sub-modules, static
        values, and structural changes like `None`.

        Examples
        --------
        >>> model = model.at.blocks[2].attn.w_q.set(new_w)  # swap a deep leaf
        >>> model = model.at.b.set(None)                    # remove bias
        >>> model = model.at.encoder.set(model.encoder.freeze())
        """
        return _At(self)

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
        params = jax.tree.map(
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
        leaves = jax.tree.leaves(self, is_leaf=tree.is_param)
        return sum(p._value.size for p in leaves if tree.is_param(p))
