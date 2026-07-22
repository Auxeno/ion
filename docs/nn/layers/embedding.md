# Embedding

Lookup table mapping integer token ids to dense vectors. Weights use fan-in variance-scaling initialization (std 1/sqrt(dim)), so each row starts near unit norm regardless of dimension and independent of vocabulary size.

::: ion.nn.Embedding
