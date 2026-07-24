Apply gradients and advance optimizer state.

Parameters
----------
model : PyTree
    Current model. Its structure and trainability must match the model used to
    construct the optimizer.
grads : PyTree
    Gradients returned by `jax.grad`.
kwargs : Any
    Additional keyword arguments forwarded to the optax transform.

Returns
-------
tuple[PyTree, Optimizer]
    Updated model and new optimizer. Neither input is mutated.

Warning
-------
Changing model structure or trainability requires constructing a new
`Optimizer`; otherwise `update` raises `ValueError`.
