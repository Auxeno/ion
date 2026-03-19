"""State space model layers and cells.

Modules:
    LRUCell  Single-step Linear Recurrent Unit with diagonal complex state.   (Orvieto et al., 2023)
    LRU      Linear Recurrent Unit over a sequence via lax.associative_scan.  (Orvieto et al., 2023)
    S4DCell  Single-step per-feature SISO S4D cell (diagonal, S4D-Lin).       (Gu et al., 2022)
    S4D      Per-feature SISO S4D over a sequence via lax.associative_scan.   (Gu et al., 2022)
    S5Cell   Single-step MIMO Simplified S5 cell with shared diagonal state.  (Smith et al., 2023)
    S5       MIMO S5 over a sequence via lax.associative_scan.                (Smith et al., 2023)

LRU: glorot normal for B/C projections, zeros for D. Eigenvalue magnitudes
initialized uniformly on complex annulus [r_min, r_max].

S4D: per-feature SISO SSMs. No explicit B parameter (B=1, absorbed into ZOH).
Glorot normal for C, zeros for D. S4D-Lin initialization for diagonal A
(-1/2 + i*pi*n). Per-feature learnable timestep.

S5: shared MIMO SSM with dense B/C projections. Glorot normal for B/C, zeros
for D. S4D-Lin initialization for diagonal A (-1/2 + i*pi*n). Per-state
learnable timestep.

All SSM hidden states are complex-valued. hidden_dim=N stores N complex values
(2N real parameters). S4D and S5 use conjugate-pair structure: only one
eigenvalue per conjugate pair is stored, and the readout uses 2*Re(...) to
recover the full contribution. LRU stores independent complex eigenvalues
without conjugate symmetry.

Input layout is (batch, time, features).
Initial state defaults to zeros if not provided.
"""

from math import pi

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


def _binary_op(a, b):
    """Binary operator for parallel scan of diagonal linear recurrence."""
    a_lambda, a_hidden = a
    b_lambda, b_hidden = b
    return b_lambda * a_lambda, b_lambda * a_hidden + b_hidden


class LRUCell(Module):
    """Single-step Linear Recurrent Unit cell.

    >>> cell = LRUCell(3, 16, key=key)
    >>> y, h = cell(x, h)  # (*, 3), (*, 16) -> (*, 3), (*, 16)
    """

    B: Param[Complex[Array, "i h"]]
    C: Param[Complex[Array, "h i"]]
    D: Param[Float[Array, " i"]]
    nu_log: Param[Float[Array, " h"]]
    theta_log: Param[Float[Array, " h"]]
    gamma_log: Param[Float[Array, " h"]]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * pi,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_b, key_c, key_d, key_nu, key_theta = jax.random.split(key, 5)

        # Complex projections: B maps input to hidden state, C maps hidden state to output
        self.B = Param(w_init(shape=(in_dim, hidden_dim), dtype=jnp.complex64, key=key_b))
        self.C = Param(w_init(shape=(hidden_dim, in_dim), dtype=jnp.complex64, key=key_c))

        # Skip connection: maps input directly to output, bypassing the recurrence
        self.D = Param(d_init(shape=(in_dim,), dtype=dtype, key=key_d))

        # Eigenvalue magnitudes sampled uniformly on annulus [r_min, r_max]
        u1 = jax.random.uniform(key_nu, shape=(hidden_dim,))
        u2 = jax.random.uniform(key_theta, shape=(hidden_dim,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        # Diagonal eigenvalues: nu decay magnitude, theta phase rotation, gamma normalization
        self.nu_log = Param(nu_log.astype(dtype))
        self.theta_log = Param(theta_log.astype(dtype))
        A = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = Param(jnp.log(jnp.sqrt(1 - jnp.abs(A) ** 2)).astype(dtype))

    @property
    def initial_state(self) -> Complex[Array, " h"]:
        return jnp.zeros(self.nu_log.shape[0], dtype=self.B.dtype)

    def __call__(
        self,
        x: Float[Array, "... i"],
        h: Complex[Array, "... h"],
    ) -> tuple[Float[Array, "... i"], Complex[Array, "... h"]]:

        A = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = self.B * jnp.exp(self.gamma_log)
        h = A * h + x.astype(self.B.dtype) @ B_norm
        x = jnp.real(h @ self.C) + self.D * x

        return x, h


class LRU(Module):
    """Linear Recurrent Unit over a full sequence.

    >>> lru = LRU(3, 16, key=key)
    >>> outputs, h = lru(x)  # (b, t, 3) -> (b, t, 3), (b, 16)
    """

    cell: LRUCell

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * pi,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = LRUCell(
            in_dim, hidden_dim, r_min, r_max, max_phase, dtype, w_init, d_init, key=key
        )

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: Complex[Array, "b h"] | None = None,
    ) -> tuple[Float[Array, "b t i"], Complex[Array, "b h"]]:

        b, t, i = x.shape

        A = jnp.exp(-jnp.exp(self.cell.nu_log) + 1j * jnp.exp(self.cell.theta_log))
        B_norm = self.cell.B * jnp.exp(self.cell.gamma_log)

        lambdas = jnp.broadcast_to(A, (b, t, self.cell.nu_log.shape[0]))
        hidden = x.astype(self.cell.B.dtype) @ B_norm

        lambdas, hidden = lax.associative_scan(fn=_binary_op, elems=(lambdas, hidden), axis=1)

        if hx is not None:
            hidden = lambdas * hx[:, None, :] + hidden

        x = jnp.real(hidden @ self.cell.C) + self.cell.D * x

        return x, hidden[:, -1, :]


class S4DCell(Module):
    """Single-step per-feature SISO S4D cell.

    >>> cell = S4DCell(3, 8, key=key)
    >>> x, h = cell(x, h)  # (*, 3), (*, 3, 8) -> (*, 3), (*, 3, 8)
    """

    A_log_re: Param[Float[Array, "i h"]]
    A_im: Param[Float[Array, "i h"]]
    C: Param[Complex[Array, "i h"]]
    D: Param[Float[Array, " i"]]
    log_dt: Param[Float[Array, " i"]]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_c, key_d, key_dt = jax.random.split(key, 3)

        # Eigenvalues at harmonics (-1/2 + i*pi*n) so each state captures a different frequency
        self.A_log_re = Param(jnp.full((in_dim, hidden_dim), jnp.log(0.5), dtype=dtype))
        self.A_im = Param(
            jnp.broadcast_to(
                (pi * jnp.arange(hidden_dim)).astype(dtype), (in_dim, hidden_dim)
            ).copy()
        )

        # C projects each feature's hidden state to a scalar output
        self.C = Param(w_init(shape=(in_dim, hidden_dim), dtype=jnp.complex64, key=key_c))

        # Skip connection
        self.D = Param(d_init(shape=(in_dim,), dtype=dtype, key=key_d))

        # Learnable timestep controlling how finely each feature samples continuous dynamics
        log_dt = jax.random.uniform(
            shape=(in_dim,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max), key=key_dt
        )
        self.log_dt = Param(log_dt.astype(dtype))

    @property
    def initial_state(self) -> Complex[Array, "i h"]:
        return jnp.zeros(self.A_log_re.shape, dtype=self.C.dtype)

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


class S4D(Module):
    """Per-feature SISO S4D over a full sequence.

    >>> s4d = S4D(3, 8, key=key)
    >>> outputs, h = s4d(x)  # (b, t, 3) -> (b, t, 3), (b, 3, 8)
    """

    cell: S4DCell

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = S4DCell(in_dim, hidden_dim, dt_min, dt_max, dtype, w_init, d_init, key=key)

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
    >>> x, h = cell(x, h)  # (*, 3), (*, 8) -> (*, 3), (*, 8)
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
        hidden_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_b, key_c, key_d, key_dt = jax.random.split(key, 4)

        # Eigenvalues at harmonics (-1/2 + i*pi*n) so each state captures a different frequency
        self.A_log_re = Param(jnp.full(hidden_dim, jnp.log(0.5), dtype=dtype))
        self.A_im = Param((pi * jnp.arange(hidden_dim)).astype(dtype))

        # Dense complex projections: B maps input to shared state, C maps state to output
        self.B = Param(w_init(shape=(in_dim, hidden_dim), dtype=jnp.complex64, key=key_b))
        self.C = Param(w_init(shape=(hidden_dim, in_dim), dtype=jnp.complex64, key=key_c))

        # Skip connection
        self.D = Param(d_init(shape=(in_dim,), dtype=dtype, key=key_d))

        # Learnable timestep controlling how finely each state samples continuous dynamics
        log_dt = jax.random.uniform(
            shape=(hidden_dim,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max), key=key_dt
        )
        self.log_dt = Param(log_dt.astype(dtype))

    @property
    def initial_state(self) -> Complex[Array, " h"]:
        return jnp.zeros(self.A_log_re.shape[0], dtype=self.B.dtype)

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


class S5(Module):
    """MIMO S5 over a full sequence.

    >>> s5 = S5(3, 8, key=key)
    >>> outputs, h = s5(x)  # (b, t, 3) -> (b, t, 3), (b, 8)
    """

    cell: S5Cell

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = S5Cell(in_dim, hidden_dim, dt_min, dt_max, dtype, w_init, d_init, key=key)

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
