Rotary positional embeddings ([Su et al., 2021](https://arxiv.org/abs/2104.09864)).

Encodes position by rotating pairs of features by an angle proportional to their position, applied to query and key vectors before attention. Relative position falls out of the dot product, and there are no learnable parameters.

Parameters
----------
theta : float, default=10000.0
    Base wavelength controlling the rotation frequencies across feature pairs.

Example
-------
```python
rope = nn.RoPE()
q = rope(q)  # (*, s, d) -> (*, s, d)
k = rope(k)  # (*, s, d) -> (*, s, d)
```
