Fixed sinusoidal positional encodings ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Adds the classic table of interleaved sines and cosines at geometrically spaced frequencies to the input, the fixed counterpart to `LearnedPositionalEmbedding`. It holds no parameters and needs no `key`, so the two are interchangeable at a call site.

Sequence length and feature dimension come from the input, and the table is rebuilt on each call rather than stored. Under `jit` it constant-folds, so there is nothing to cache and no maximum length to declare.

Sine and cosine alternate down the feature axis, pairing `(sin, cos)` at each frequency, as written in the paper and as `RoPE` pairs its features. Some implementations instead put every sine in the first half of the features and every cosine in the second, which changes which feature carries which phase and matters only when porting weights trained against it.

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

# Swap for a trained table without touching the call site
pos = nn.LearnedPositionalEmbedding(128, 64, key=key)
y = pos(x)  # (4, 128, 64) -> (4, 128, 64)
```
