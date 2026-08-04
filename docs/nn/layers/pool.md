# Pooling

N-dimensional pooling. Like convolution, spatial rank is inferred from `kernel_shape`, so the same class handles 1D, 2D, and 3D. These layers hold no parameters and take no `key`.

Pooling is computed in `float32` for numerical stability, then cast back to the input dtype.

::: ion.nn.MaxPool

::: ion.nn.AvgPool
