# Stochastic regularization

`Dropout` provides element-wise and broadcast masks. Broadcast a mask over a residual branch to express stochastic depth. Training mode is explicit, and training calls take a random `key` when `p > 0`.

::: ion.nn.Dropout
