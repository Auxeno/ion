"""Lightweight wrapper that marks arrays as mutable model state.

Classes:
    Buffer  Marks a JAX array as a non-trainable value updated in place.

Registered as a JAX pytree with no children: the underlying reference is static
metadata, so buffers stay invisible to `jax.grad`, `ion.Optimizer` and `astype`.
Writes go through `set`, which applies `jax.lax.stop_gradient`.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from .param import Param


@jtu.register_pytree_node_class
class Buffer:
    """Marks a JAX array as a non-trainable value updated in place.

    Parameters
    ----------
    value : Array
        Initial value of the buffer.

    Notes
    -----
    Buffers hold state such as BatchNorm running statistics. They are updated
    during a forward pass rather than by an optimizer, and `set` applies
    `stop_gradient` so they stay out of autodiff.
    A buffer is mutable, so copies sharing one buffer share its state.

    Examples
    --------
    >>> running_mean = Buffer(jnp.zeros(16))
    >>> running_mean.set(0.9 * running_mean.value + 0.1 * mean)
    """

    __slots__ = ("_ref",)

    def __init__(self, value: Array) -> None:
        self._ref = jax.new_ref(jnp.asarray(value))

    @property
    def value(self) -> Array:
        """Current value of the buffer."""
        return self._ref[...]

    def set(self, value: Array) -> None:
        """Replace the stored value, applying `stop_gradient`."""
        self._ref[...] = jax.lax.stop_gradient(value)

    def tree_flatten(self) -> tuple[tuple, jax.Ref]:
        return (), self._ref

    @classmethod
    def tree_unflatten(cls, aux: jax.Ref, children: tuple) -> "Buffer":
        """Rebuild around the existing reference, so copies keep sharing one buffer."""
        buffer = object.__new__(cls)
        buffer._ref = aux
        return buffer

    def __repr__(self) -> str:
        value = self.value
        return f"Buffer({Param.short_dtype(value.dtype.name)}{list(value.shape)})"

    def __treescope_repr__(self, path: str | None, subtree_renderer: Any) -> Any:
        """Hook to make `Buffer`s colored in grey in Treescope."""
        import treescope

        return treescope.repr_lib.render_object_constructor(
            object_type=type(self),
            attributes={"value": self.value},
            path=path,
            subtree_renderer=subtree_renderer,
            color="oklch(0.925 0.0 0.0)",
        )
