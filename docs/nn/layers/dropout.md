# Dropout

Stochastic dropout. Because Ion layers are stateless, `Dropout` takes a `key` at call time for the random mask rather than reading a global training flag.

::: ion.nn.Dropout
