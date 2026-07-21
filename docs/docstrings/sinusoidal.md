Fixed sinusoidal positional encodings (Vaswani et al., 2017).

Builds the classic table of interleaved sines and cosines at geometrically spaced frequencies. Add the result to input features; it holds no parameters and needs no `key`.

Parameters
----------
seq_len : int
    Number of positions (rows).
dim : int
    Feature dimension (columns). Interleaves sine and cosine pairs, so even
    values are typical.
theta : float, default=10000.0
    Base wavelength controlling the frequency spacing across feature pairs.
dtype : jnp.dtype, default=jnp.float32
    Dtype of the returned array.

Returns
-------
jax.Array["seq_len dim", float]
    Positional encoding table.

Examples
--------
>>> pe = nn.sinusoidal(128, 64)  # (128, 64)
>>> y = x + pe                    # add to (*, 128, 64) features
