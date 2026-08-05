Fixed sinusoidal positional encodings ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Parameters
----------
theta : float, default=10000.0
    Base wavelength controlling the frequency spacing across feature pairs.

Example
-------
```python
pos = nn.SinusoidalPositionalEmbedding()
x = jnp.ones((4, 128, 64))
y = pos(x)  # (4, 128, 64) -> (4, 128, 64)
```
