# Sequential

Chains a list of layers (or plain callables) into one module, applying them in order and forwarding an optional `key` to layers that take one. A composite module, but constructed and called like any other layer.

::: ion.nn.Sequential
