# Blocks

Composite modules that assemble layers into reusable pieces. `MLP` builds a multi-layer perceptron from a list of dimensions; `Sequential` chains a list of layers (or callables) into one module. Both are ordinary `Module`s, so they compose and transform like any other layer.

::: ion.nn.MLP

::: ion.nn.Sequential
