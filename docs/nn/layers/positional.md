# Positional

Positional encodings for sequence models. `RoPE` rotates query and key features by position; `LearnedPositionalEmbedding` adds a trained per-position vector; `sinusoidal` and `alibi` are functions that build fixed encodings (an additive sinusoidal table and an attention-bias slope, respectively).

::: ion.nn.RoPE

::: ion.nn.LearnedPositionalEmbedding

::: ion.nn.sinusoidal

::: ion.nn.alibi
