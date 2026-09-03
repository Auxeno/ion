# Pooling

N-dimensional pooling. Like convolution, spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, and 3D. These layers hold no parameters and take no `key`.

`AvgPool` accumulates in `float32` for numerical stability, then casts back to the input dtype. `MaxPool` reduces in the input dtype, since a maximum is exact in any precision.

::: ion.nn.MaxPool

::: ion.nn.AvgPool
