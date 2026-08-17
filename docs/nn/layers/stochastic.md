# Stochastic regularization

`Dropout` drops individual elements, and `DropPath` drops a residual branch for whole samples to give stochastic depth. Both take a drop probability alone. Training mode is explicit, and training calls take a random `key` when `p > 0`.

::: ion.nn.Dropout

::: ion.nn.DropPath
