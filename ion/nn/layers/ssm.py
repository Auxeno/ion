"""State space model layers and cells.

Modules:
    S4DCell  Single-step per-feature SISO S4D cell (diagonal, S4D-Lin).       (Gu et al., 2022)
    S4D      Per-feature SISO S4D over a sequence via lax.associative_scan.   (Gu et al., 2022)
    S5Cell   Single-step MIMO S5 cell with shared diagonal state.             (Smith et al., 2023)
    S5       MIMO S5 over a sequence via lax.associative_scan.                (Smith et al., 2023)

Sequence layers use associative scan for O(log T) parallel time complexity.
All hidden states are complex-valued. S4D and S5 use conjugate pairs
(state_dim N stores N//2 eigenvalues, readout via 2*Re).
Glorot uniform for projections, zeros for D and skip connections.
Input layout is (batch, time, features).
Initial state defaults to zeros if not provided.
"""

from math import pi

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer, glorot_uniform, zeros

from ...typing import Array, Complex, Float, PRNGKey
from ..module import Module
from ..param import Param


def _binary_op(a: tuple[Array, Array], b: tuple[Array, Array]) -> tuple[Array, Array]:
    """Binary operator for parallel scan of diagonal linear recurrence."""
    a_lambda, a_hidden = a
    b_lambda, b_hidden = b
    return b_lambda * a_lambda, b_lambda * a_hidden + b_hidden


class S4DCell(Module):
    """Single-step per-feature SISO S4D cell.

    >>> cell = S4DCell(3, 8, key=key)
    >>> y, h = cell(x, h)  # (*, 3), (*, 3, 4) -> (*, 3), (*, 3, 4)
    """

    A_log_re: Param[Float[Array, "i h"]]
    A_im: Param[Float[Array, "i h"]]
    C: Param[Complex[Array, "i h"]]
    D: Param[Float[Array, " i"]]
    log_dt: Param[Float[Array, " i"]]

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        w_init: Initializer = glorot_uniform(),
        d_init: Initializer = zeros,
        key: PRNGKey,
    ) -> None:

        if state_dim < 2 or state_dim % 2 != 0:
            raise ValueError(f"state_dim ({state_dim}) must be a positive even number")

        # Halve for conjugate pairs
        h = state_dim // 2

        key_c, key_d, key_dt = jax.random.split(key, 3)

        # Eigenvalues at harmonics (-1/2 + i*pi*n) so each state captures a different frequency
        self.A_log_re = Param(jnp.full((in_dim, h), jnp.log(0.5)))
        self.A_im = Param(jnp.broadcast_to(pi * jnp.arange(h), (in_dim, h)).copy())

        # C projects each feature's hidden state to a scalar output
        self.C = Param(w_init(shape=(in_dim, h), dtype=jnp.complex64, key=key_c))

        # Skip connection
        self.D = Param(d_init(shape=(in_dim,), key=key_d))

        # Learnable timestep controlling how finely each feature samples continuous dynamics
        log_dt = jax.random.uniform(
            key_dt, shape=(in_dim,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
        )
        self.log_dt = Param(log_dt)

    def __call__(
        self,
        x: Float[Array, "... i"],
        h: Complex[Array, "... i h"],
    ) -> tuple[Float[Array, "... i"], Complex[Array, "... i h"]]:

        dt = jnp.exp(self.log_dt)
        A = -jnp.exp(self.A_log_re) + 1j * self.A_im
        A_bar = jnp.exp(A * dt[:, None])

        # Input-to-state gain after discretization (B=1, each input drives its own states)
        B_bar = (A_bar - 1.0) / A

        h = A_bar * h + B_bar * x[..., :, None].astype(self.C.dtype)

        # 2*Re recovers full output from half the conjugate eigenvalue pairs
        x = 2.0 * jnp.real(jnp.sum(self.C * h, axis=-1)) + self.D * x

        return x, h

    @property
    def initial_state(self) -> Complex[Array, "i h"]:
        return jnp.zeros(self.A_log_re.shape, dtype=self.C.dtype)


class S4D(Module):
    """Per-feature SISO S4D over a full sequence.

    >>> s4d = S4D(3, 8, key=key)
    >>> outputs, h = s4d(x)  # (b, t, 3) -> (b, t, 3), (b, 3, 4)
    """

    cell: S4DCell

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        w_init: Initializer = glorot_uniform(),
        d_init: Initializer = zeros,
        key: PRNGKey,
    ) -> None:

        self.cell = S4DCell(
            in_dim, state_dim, dt_min=dt_min, dt_max=dt_max, w_init=w_init, d_init=d_init, key=key
        )

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: Complex[Array, "b i h"] | None = None,
    ) -> tuple[Float[Array, "b t i"], Complex[Array, "b i h"]]:

        b, t, i = x.shape

        dt = jnp.exp(self.cell.log_dt)
        A = -jnp.exp(self.cell.A_log_re) + 1j * self.cell.A_im
        A_bar = jnp.exp(A * dt[:, None])
        B_bar = (A_bar - 1.0) / A

        lambdas = jnp.broadcast_to(A_bar, (b, t, *self.cell.A_log_re.shape))
        hidden = B_bar * x[..., None].astype(self.cell.C.dtype)

        lambdas, hidden = lax.associative_scan(fn=_binary_op, elems=(lambdas, hidden), axis=1)

        if hx is not None:
            hidden = lambdas * hx[:, None, :, :] + hidden

        x = 2.0 * jnp.real(jnp.sum(self.cell.C * hidden, axis=-1)) + self.cell.D * x

        return x, hidden[:, -1, :, :]


class S5Cell(Module):
    """Single-step MIMO S5 cell with shared diagonal state.

    >>> cell = S5Cell(3, 8, key=key)
    >>> y, h = cell(x, h)  # (*, 3), (*, 4) -> (*, 3), (*, 4)
    """

    A_log_re: Param[Float[Array, " h"]]
    A_im: Param[Float[Array, " h"]]
    B: Param[Complex[Array, "i h"]]
    C: Param[Complex[Array, "h i"]]
    D: Param[Float[Array, " i"]]
    log_dt: Param[Float[Array, " h"]]

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        w_init: Initializer = glorot_uniform(),
        d_init: Initializer = zeros,
        key: PRNGKey,
    ) -> None:

        if state_dim < 2 or state_dim % 2 != 0:
            raise ValueError(f"state_dim ({state_dim}) must be a positive even number")

        # Halve for conjugate pairs
        h = state_dim // 2

        key_b, key_c, key_d, key_dt = jax.random.split(key, 4)

        # Eigenvalues at harmonics (-1/2 + i*pi*n) so each state captures a different frequency
        self.A_log_re = Param(jnp.full(h, jnp.log(0.5)))
        self.A_im = Param(pi * jnp.arange(h))

        # Dense complex projections: B maps input to shared state, C maps state to output
        self.B = Param(w_init(shape=(in_dim, h), dtype=jnp.complex64, key=key_b))
        self.C = Param(w_init(shape=(h, in_dim), dtype=jnp.complex64, key=key_c))

        # Skip connection
        self.D = Param(d_init(shape=(in_dim,), key=key_d))

        # Learnable timestep controlling how finely each state samples continuous dynamics
        log_dt = jax.random.uniform(
            key_dt, shape=(h,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
        )
        self.log_dt = Param(log_dt)

    def __call__(
        self,
        x: Float[Array, "... i"],
        h: Complex[Array, "... h"],
    ) -> tuple[Float[Array, "... i"], Complex[Array, "... h"]]:

        dt = jnp.exp(self.log_dt)
        A = -jnp.exp(self.A_log_re) + 1j * self.A_im
        A_bar = jnp.exp(A * dt)

        # Discretized input projection (zero-order hold)
        B_bar = self.B * ((A_bar - 1.0) / A)

        h = A_bar * h + x.astype(self.B.dtype) @ B_bar

        # 2*Re recovers full output from half the conjugate eigenvalue pairs
        x = 2.0 * jnp.real(h @ self.C) + self.D * x

        return x, h

    @property
    def initial_state(self) -> Complex[Array, " h"]:
        return jnp.zeros(self.A_log_re.shape[0], dtype=self.B.dtype)


class S5(Module):
    """MIMO S5 over a full sequence.

    >>> s5 = S5(3, 8, key=key)
    >>> outputs, h = s5(x)  # (b, t, 3) -> (b, t, 3), (b, 4)
    """

    cell: S5Cell

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        *,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        w_init: Initializer = glorot_uniform(),
        d_init: Initializer = zeros,
        key: PRNGKey,
    ) -> None:

        self.cell = S5Cell(
            in_dim, state_dim, dt_min=dt_min, dt_max=dt_max, w_init=w_init, d_init=d_init, key=key
        )

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: Complex[Array, "b h"] | None = None,
    ) -> tuple[Float[Array, "b t i"], Complex[Array, "b h"]]:

        b, t, i = x.shape

        dt = jnp.exp(self.cell.log_dt)
        A = -jnp.exp(self.cell.A_log_re) + 1j * self.cell.A_im
        A_bar = jnp.exp(A * dt)
        B_bar = self.cell.B * ((A_bar - 1.0) / A)

        lambdas = jnp.broadcast_to(A_bar, (b, t, self.cell.A_log_re.shape[0]))
        hidden = x.astype(self.cell.B.dtype) @ B_bar

        lambdas, hidden = lax.associative_scan(fn=_binary_op, elems=(lambdas, hidden), axis=1)

        if hx is not None:
            hidden = lambdas * hx[:, None, :] + hidden

        x = 2.0 * jnp.real(hidden @ self.cell.C) + self.cell.D * x

        return x, hidden[:, -1, :]
