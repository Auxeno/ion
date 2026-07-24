Stochastic dropout ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)).

Randomly zeros elements with probability `p` and scales the survivors by `1 / (1 - p)` (inverted dropout), so activation magnitudes match between training and inference.

Parameters
----------
p : float
    Drop probability in `[0, 1)`.
deterministic : bool, default=False
    If `True`, the layer is a no-op (used for eval). Can be overridden per call.

Example
-------
```python
drop = nn.Dropout(0.5)
x = jnp.ones((8, 64))
y = drop(x, key=key)  # (8, 64), mask sampled from key
y = drop(x, deterministic=True)  # (8, 64), pass-through
```
