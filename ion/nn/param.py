"""Lightweight wrapper that marks arrays as model parameters.

Classes:
    Param   Marks a JAX array as trainable or frozen.

Registered as a JAX pytree: `_value` is a dynamic child, `trainable` is static metadata.
Implements `__jax_array__` and arithmetic so it works as a drop-in for plain arrays.
Setting `trainable=False` applies `jax.lax.stop_gradient` inside `__jax_array__`,
making the parameter invisible to autodiff.

See docs/internals.md for implementation details.
"""

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

T = TypeVar("T", bound=Array)

if TYPE_CHECKING:

    class _ParamBase(jax.Array, Generic[T]): ...
else:

    class _ParamBase(Generic[T]): ...


def _unwrap(x: Any) -> Any:
    """Extract the underlying array from a `Param`, or pass through as-is."""
    return jnp.asarray(x) if isinstance(x, Param) else x


@functools.partial(jtu.register_dataclass, data_fields=["_value"], meta_fields=["trainable"])
@dataclasses.dataclass(frozen=True, eq=False)
class Param(_ParamBase[T]):
    """Marks a JAX array as a model parameter.

    Parameters
    ----------
    _value : Array
        The underlying JAX array.
    trainable : bool, optional
        Whether the parameter is trainable (default `True`).

    Notes
    -----
    Frozen params have `stop_gradient` applied via `__jax_array__`.
    Arithmetic ops return plain arrays, not `Param` instances.
    Array attributes (`.shape`, `.dtype`, etc.) are proxied to the underlying array.

    Examples
    --------
    >>> w = Param(jnp.zeros(16))                 # trainable by default
    >>> b = Param(jnp.ones(4), trainable=False)  # frozen
    """

    _value: T
    trainable: bool = True

    def __jax_array__(self) -> Array:
        if self.trainable:
            return self._value
        return jax.lax.stop_gradient(self._value)

    def __getattr__(self, name: str) -> Any:
        # Do not forward explicit dunder retrieval
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.__jax_array__(), name)

    def __getitem__(self, key: Any) -> Array:
        return jnp.asarray(self)[key]

    def __add__(self, other: Any) -> Array:
        return jnp.asarray(self) + _unwrap(other)

    def __radd__(self, other: Any) -> Array:
        return _unwrap(other) + jnp.asarray(self)

    def __sub__(self, other: Any) -> Array:
        return jnp.asarray(self) - _unwrap(other)

    def __rsub__(self, other: Any) -> Array:
        return _unwrap(other) - jnp.asarray(self)

    def __mul__(self, other: Any) -> Array:
        return jnp.asarray(self) * _unwrap(other)

    def __rmul__(self, other: Any) -> Array:
        return _unwrap(other) * jnp.asarray(self)

    def __truediv__(self, other: Any) -> Array:
        return jnp.asarray(self) / _unwrap(other)

    def __rtruediv__(self, other: Any) -> Array:
        return _unwrap(other) / jnp.asarray(self)

    def __floordiv__(self, other: Any) -> Array:
        return jnp.asarray(self) // _unwrap(other)

    def __rfloordiv__(self, other: Any) -> Array:
        return _unwrap(other) // jnp.asarray(self)

    def __mod__(self, other: Any) -> Array:
        return jnp.asarray(self) % _unwrap(other)

    def __rmod__(self, other: Any) -> Array:
        return _unwrap(other) % jnp.asarray(self)

    def __pow__(self, other: Any) -> Array:
        return jnp.asarray(self) ** _unwrap(other)

    def __rpow__(self, other: Any) -> Array:
        return _unwrap(other) ** jnp.asarray(self)

    def __matmul__(self, other: Any) -> Array:
        return jnp.asarray(self) @ _unwrap(other)

    def __rmatmul__(self, other: Any) -> Array:
        return _unwrap(other) @ jnp.asarray(self)

    def __neg__(self) -> Array:
        return -jnp.asarray(self)

    def __pos__(self) -> Array:
        return +jnp.asarray(self)

    def __abs__(self) -> Array:
        return abs(jnp.asarray(self))

    def __eq__(self, other: Any) -> Array:
        return jnp.asarray(self) == _unwrap(other)

    def __ne__(self, other: Any) -> Array:
        return jnp.asarray(self) != _unwrap(other)

    def __lt__(self, other: Any) -> Array:
        return jnp.asarray(self) < _unwrap(other)

    def __le__(self, other: Any) -> Array:
        return jnp.asarray(self) <= _unwrap(other)

    def __gt__(self, other: Any) -> Array:
        return jnp.asarray(self) > _unwrap(other)

    def __ge__(self, other: Any) -> Array:
        return jnp.asarray(self) >= _unwrap(other)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self):
        return iter(self._value)

    @staticmethod
    def short_dtype(name: str) -> str:
        """Abbreviate dtype names: float32 to f32, bfloat16 to bf16, etc."""
        for prefix, abbr in (
            ("bfloat", "bf"),
            ("float", "f"),
            ("uint", "u"),
            ("int", "i"),
            ("complex", "c"),
        ):
            if name.startswith(prefix):
                return abbr + name[len(prefix) :]
        return name

    def __repr__(self) -> str:
        trainable_str = f", trainable={self.trainable}"
        if hasattr(self._value, "shape") and hasattr(self._value, "dtype"):
            dtype = self.short_dtype(self._value.dtype.name)
            return f"Param({dtype}{list(self._value.shape)}{trainable_str})"
        return f"Param({self._value!r}{trainable_str})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to make `Param`s colored in grey in Treescope."""
        import treescope

        attributes = {"value": self._value, "trainable": self.trainable}

        # Grey for trainable, ice blue for non-trainable/frozen
        color = "oklch(0.92 0.06 260.0)" if not self.trainable else "oklch(0.925 0.0 0.0)"

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=attributes,
            path=path,
            subtree_renderer=subtree_renderer,
            color=color,
        )
