Run sequence layers in both time directions.

Parameters
----------
forward : Module
    Sequence layer for the forward direction.
backward : Module
    Independent sequence layer for the backward direction.
mode : str, default='sum'
    Output combination: `"sum"`, `"concat"`, or `"mean"`.

Attributes
----------
forward : Module
    Forward sequence layer.
backward : Module
    Backward sequence layer.
mode : str
    Output combination mode.

Example
-------
```python
forward_key, backward_key = jax.random.split(key)
layer = nn.Bidirectional(
    nn.LSTM(3, 16, key=forward_key),
    nn.LSTM(3, 16, key=backward_key),
)
x = jnp.ones((8, 20, 3))
outputs, states = layer(x)  # (8, 20, 16), two final states
```
