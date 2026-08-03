Stochastic dropout ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)).

Randomly zeros elements with probability `p` and scales the survivors by
`1 / (1 - p)` (inverted dropout), so activation magnitudes match between
training and inference.

Parameters
----------
p : float
    Drop probability in `[0, 1]`.

Example
-------
```python
drop = nn.Dropout(0.5)
x = jnp.ones((8, 64))
y = drop(x, training=True, key=key)  # (8, 64), mask sampled from key
y = drop(x, training=False)  # (8, 64), pass-through
```
