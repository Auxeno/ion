"""Lightweight wrapper that marks arrays as model parameters.

Classes:
    Param   Marks a JAX array as trainable or frozen.

Registered as a JAX pytree: `value` is a dynamic child, `trainable` is static metadata.
Implements `__jax_array__` and arithmetic so it works as a drop-in for plain arrays.

See docs/internals.md for implementation details.
"""

import dataclasses
import functools
from typing import Any, Generic, TypeVar

import jax.tree_util as jtu
from jaxtyping import Array

T = TypeVar("T", bound=Array)


def _unwrap(x: Any) -> Any:
    """Extract the underlying array from a `Param`, or pass through as-is."""
    return x.value if isinstance(x, Param) else x


@functools.partial(jtu.register_dataclass, data_fields=["value"], meta_fields=["trainable"])
@dataclasses.dataclass(frozen=True, eq=False)
class Param(Generic[T]):
    """Marks a JAX array as a model parameter.

    >>> w = Param(jnp.zeros(16))                 # trainable by default
    >>> b = Param(jnp.ones(4), trainable=False)  # frozen
    """

    value: T
    trainable: bool = True

    def __jax_array__(self) -> Array:
        return self.value

    def __getattr__(self, name: str) -> Any:
        # Do not forward explicit dunder retrieval
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.value, name)

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

    def __bool__(self) -> bool:
        return bool(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __repr__(self) -> str:
        trainable_str = f", trainable={self.trainable}"
        if hasattr(self.value, "shape") and hasattr(self.value, "dtype"):
            dtype = {
                "float16": "f16",
                "float32": "f32",
                "float64": "f64",
                "bfloat16": "bf16",
                "int8": "i8",
                "int16": "i16",
                "int32": "i32",
                "int64": "i64",
                "uint8": "u8",
                "uint16": "u16",
                "uint32": "u32",
                "uint64": "u64",
                "complex64": "c64",
                "complex128": "c128",
            }.get(self.value.dtype.name, self.value.dtype.name)
            return f"Param({dtype}{list(self.value.shape)}{trainable_str})"
        return f"Param({self.value!r}{trainable_str})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to make `Param`s colored in grey in Treescope."""
        import treescope

        attributes = {"value": self.value, "trainable": self.trainable}

        # Grey for trainable, ice blue for non-trainable/frozen
        color = "oklch(0.9 0.03 260.0)" if not self.trainable else "oklch(0.925 0.0 0.0)"

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes=attributes,
            path=path,
            subtree_renderer=subtree_renderer,
            color=color,
        )
