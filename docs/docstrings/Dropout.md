Stochastic dropout (Srivastava et al., 2014).

Randomly zeros elements with probability `p` and scales the survivors by `1 / (1 - p)` (inverted dropout), so activation magnitudes match between training and inference.

Parameters
----------
p : float
    Drop probability in `[0, 1)`.
deterministic : bool, default=False
    If `True`, the layer is a no-op (used for eval). Can be overridden per call.

Notes
-----
Ion layers are stateless, so `Dropout` reads no global training flag. Pass a `key` at call time to sample the mask; omit it (or set `deterministic=True`) to pass the input through unchanged. A `key` is required unless the call is deterministic.

Examples
--------
>>> drop = nn.Dropout(0.5)
>>> y = drop(x, key=key)                  # training: mask sampled from key
>>> y = drop(x, deterministic=True)       # eval: pass-through, no key needed
