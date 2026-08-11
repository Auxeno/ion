Stochastic dropout ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)).

Randomly zeros elements with probability `p` and scales the survivors by
`1 / (1 - p)` (inverted dropout), so activation magnitudes match between
training and inference. Dimensions listed in `broadcast_dims` share the same
mask values.

Parameters
----------
p : float
    Drop probability in `[0, 1]`.
broadcast_dims : tuple[int, ...], default=()
    Input dimensions across which to share mask values.

Example
-------
```python
drop = nn.Dropout(0.5)
x = jnp.ones((8, 64))
y = drop(x, training=True, key=key)  # (8, 64), mask sampled from key
y = drop(x, training=False)  # (8, 64), pass-through
```

Stochastic depth is dropout shared across every non-batch dimension of a
residual branch. For an MLP input shaped `(batch, features)`:

```python
drop_path = nn.Dropout(0.1, broadcast_dims=(1,))
branch = mlp(x)
x = x + drop_path(branch, training=True, key=key)
```

For a channels-last CNN stack shaped `(batch, height, width, channels)`:

```python
drop_path = nn.Dropout(0.1, broadcast_dims=(1, 2, 3))
branch = conv_2(jax.nn.relu(conv_1(x)))
x = x + drop_path(branch, training=True, key=key)
```

Each batch item keeps or drops its whole branch independently. The branch is
still evaluated before its output is masked.
