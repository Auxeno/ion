Pass-through layer that returns its first argument unchanged and ignores the rest.

Useful as a placeholder for optional layers (e.g. a normalization slot that can be switched off) or as a residual stub, so surrounding code needs no conditional.

Examples
--------
>>> layer = nn.Identity()
>>> y = layer(x)  # x unchanged
