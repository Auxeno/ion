"""Lightweight wrapper that marks arrays as model parameters.

Classes:
    Param   Marks a JAX array as trainable or frozen.

Registered as a JAX pytree: `_value` is a dynamic child, `trainable` is static metadata.
Implements `__jax_array__` and arithmetic so it works as a drop-in for plain arrays.
Setting `trainable=False` applies `jax.lax.stop_gradient` inside `__jax_array__`,
making the parameter invisible to autodiff.
"""

import dataclasses
import functools
from collections.abc import Iterator
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
    return x.value if isinstance(x, Param) else x


@functools.partial(jtu.register_dataclass, data_fields=["_value"], meta_fields=["trainable"])
@dataclasses.dataclass(frozen=True, eq=False)
class Param(_ParamBase[T]):
    """Marks a JAX array as a model parameter.

    Parameters
    ----------
    _value : Array
        Raw stored array. Read `value` instead, which respects `stop_gradient`.
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
    >>> w.value                                  # underlying array
    """

    _value: T
    trainable: bool = True

    @property
    def value(self) -> Array:
        """The parameter as autodiff sees it, with `stop_gradient` applied if frozen."""
        return jnp.asarray(self)

    def __jax_array__(self) -> Array:
        return self._value if self.trainable else jax.lax.stop_gradient(self._value)

    def __getattr__(self, name: str) -> Any:
        # Do not forward explicit dunder retrieval
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.__jax_array__(), name)

    def __getitem__(self, key: Any) -> Array:
        return self.value[key]

    def __add__(self, other: Any) -> Array:
        return self.value + _unwrap(other)

    def __radd__(self, other: Any) -> Array:
        return _unwrap(other) + self.value

    def __sub__(self, other: Any) -> Array:
        return self.value - _unwrap(other)

    def __rsub__(self, other: Any) -> Array:
        return _unwrap(other) - self.value

    def __mul__(self, other: Any) -> Array:
        return self.value * _unwrap(other)

    def __rmul__(self, other: Any) -> Array:
        return _unwrap(other) * self.value

    def __truediv__(self, other: Any) -> Array:
        return self.value / _unwrap(other)

    def __rtruediv__(self, other: Any) -> Array:
        return _unwrap(other) / self.value

    def __floordiv__(self, other: Any) -> Array:
        return self.value // _unwrap(other)

    def __rfloordiv__(self, other: Any) -> Array:
        return _unwrap(other) // self.value

    def __mod__(self, other: Any) -> Array:
        return self.value % _unwrap(other)

    def __rmod__(self, other: Any) -> Array:
        return _unwrap(other) % self.value

    def __pow__(self, other: Any) -> Array:
        return self.value ** _unwrap(other)

    def __rpow__(self, other: Any) -> Array:
        return _unwrap(other) ** self.value

    def __matmul__(self, other: Any) -> Array:
        return self.value @ _unwrap(other)

    def __rmatmul__(self, other: Any) -> Array:
        return _unwrap(other) @ self.value

    def __neg__(self) -> Array:
        return -self.value

    def __pos__(self) -> Array:
        return +self.value

    def __abs__(self) -> Array:
        return abs(self.value)

    def __eq__(self, other: Any) -> Array:
        return self.value == _unwrap(other)

    def __ne__(self, other: Any) -> Array:
        return self.value != _unwrap(other)

    def __lt__(self, other: Any) -> Array:
        return self.value < _unwrap(other)

    def __le__(self, other: Any) -> Array:
        return self.value <= _unwrap(other)

    def __gt__(self, other: Any) -> Array:
        return self.value > _unwrap(other)

    def __ge__(self, other: Any) -> Array:
        return self.value >= _unwrap(other)

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._value)

    def __repr__(self) -> str:
        frozen = "" if self.trainable else ", frozen"
        if hasattr(self._value, "dtype"):
            return f"Param({self._value.dtype.name}{self._value.shape}{frozen})"
        return f"Param({self._value!r}{frozen})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to render with Treescope."""
        from .. import _treescope

        return _treescope.param(self, path, subtree_renderer)
