Zero initial hidden state for one sequence.

Returns
-------
jax.Array["h", float]
    Unbatched zero state of shape `(hidden_dim,)`, with the same dtype as the
    recurrent weights.
