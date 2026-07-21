Learnable absolute positional embeddings.

Adds a trained per-position vector to the input, the scheme used by GPT-2 and BERT. Position `i` adds row `i` of the table.

Parameters
----------
max_len : int
    Maximum sequence length. Inputs longer than this are unsupported.
dim : int
    Feature dimension, matching the input's last dimension.
w_init : Initializer
    Weight initializer. Truncated normal (std 0.02) by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Positional table of shape `(max_len, dim)`.

Notes
-----
Only the first `s` rows are used for a length-`s` input, so the same layer serves any length up to `max_len`. Unlike `RoPE` and `sinusoidal`, positions are learned rather than fixed.

Examples
--------
>>> pos = nn.LearnedPositionalEmbedding(128, 64, key=key)
>>> y = pos(x)  # (*, s, 64) -> (*, s, 64), s <= 128
