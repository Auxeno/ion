Replace the stored value.

Parameters
----------
value : jax.Array
    New value, of the same shape and dtype as the current one.

Info
----
`jax.lax.stop_gradient` is applied to the new value, so a buffer update never
contributes to parameter gradients.
