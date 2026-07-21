Marks a JAX array as a model parameter, making the trainable/frozen distinction explicit inside a pytree.

Parameters
----------
_value : Array
    The underlying JAX array.
trainable : bool, default=True
    Whether gradients flow to this parameter. Frozen params (`trainable=False`)
    have `jax.lax.stop_gradient` applied via `__jax_array__`, so they are
    invisible to autodiff and allocated no optimizer state.

Notes
-----
Array attributes (`.shape`, `.dtype`, `.T`, ...) and `jnp` operations proxy to the underlying array through `__jax_array__`, so a `Param` is usable anywhere a raw array is. Arithmetic returns plain arrays, not `Param` instances. The `_value` field is private: reach the raw array with `jnp.asarray(param)`, never `param._value`, so `stop_gradient` is preserved for frozen params.

Examples
--------
>>> w = Param(jnp.zeros((3, 16)))              # trainable by default
>>> b = Param(jnp.zeros(16), trainable=False)  # frozen
>>> w.shape                                    # attributes proxy through
(3, 16)
