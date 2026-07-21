Rotary positional embeddings (Su et al., 2021).

Encodes position by rotating pairs of features by an angle proportional to their position, applied to query and key vectors before attention. Relative position falls out of the dot product, and there are no learnable parameters.

Parameters
----------
theta : float, default=10000.0
    Base wavelength controlling the rotation frequencies across feature pairs.

Notes
-----
Apply to queries and keys (not values), after splitting into heads, so the last dimension is the per-head dimension, which must be even. `theta` is stored as static config, not a trainable parameter.

Examples
--------
>>> rope = nn.RoPE()
>>> q = rope(q)  # (*, s, d) -> (*, s, d)
>>> k = rope(k)  # (*, s, d) -> (*, s, d)
