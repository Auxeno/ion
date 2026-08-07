"""Lightweight wrapper that marks arrays as mutable model state.

Classes:
    Buffer  Marks a JAX array as a non-trainable value updated in place.

Registered as a JAX pytree with no children: the underlying reference is static
metadata, so buffers stay invisible to `jax.grad`, `ion.Optimizer` and `astype`.
Writes go through `set`, which applies `jax.lax.stop_gradient`.
"""

from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import Tracer
from jaxtyping import Array

from .param import Param

T = TypeVar("T", bound=Array)


@jtu.register_pytree_node_class
class Buffer(Generic[T]):
    """Holds persistent, non-trainable array state mutated by a forward pass.

    Parameters
    ----------
    value : Array
        Initial value of the buffer.

    Notes
    -----
    Buffers are for state such as BatchNorm running statistics: values owned by
    a layer and updated during its forward pass rather than by an optimizer.
    `set` applies `stop_gradient` so buffer state stays out of autodiff.
    A buffer is mutable, so copies sharing one buffer share its state.
    Use an ordinary JAX array for non-parameter data that does not require mutation.

    Examples
    --------
    >>> running_mean = Buffer(jnp.zeros(16))
    >>> running_mean.set(jnp.ones(16))
    >>> running_mean.value.shape
    (16,)
    """

    __slots__ = ("_ref",)

    def __init__(self, value: T) -> None:
        array = jnp.asarray(value)
        if isinstance(array, Tracer):
            raise ValueError("Cannot create a Buffer inside a JAX transform, build models eagerly")
        self._ref = jax.new_ref(array)

    @property
    def value(self) -> T:
        """Current value of the buffer."""
        return self._ref[...]

    def set(self, value: T) -> None:
        """Replace the stored value, applying `stop_gradient`."""
        self._ref[...] = jax.lax.stop_gradient(value)

    def tree_flatten(self) -> tuple[tuple, jax.Ref]:
        return (), self._ref

    @classmethod
    def tree_unflatten(cls, aux: jax.Ref, children: tuple) -> "Buffer[T]":
        """Rebuild around the existing reference, so copies keep sharing one buffer."""
        buffer = object.__new__(cls)
        buffer._ref = aux
        return buffer

    def __repr__(self) -> str:
        value = self.value
        return f"Buffer({Param.short_dtype(value.dtype.name)}{list(value.shape)})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to render `Buffer`s on a single line in Treescope."""
        from treescope import rendering_parts as parts

        # An extra abbreviation level hides the array statistics until the Buffer is expanded
        array = parts.abbreviation_level(subtree_renderer(self.value, path=None).renderable)

        return parts.build_foldable_tree_node_from_children(
            prefix="Buffer(",
            children=[array],
            suffix=")",
            path=path,
            expand_state=parts.ExpandState.COLLAPSED,
        )
