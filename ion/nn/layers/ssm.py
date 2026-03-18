"""State space model layers and cells.

Modules:
    LRUCell  Single-step Linear Recurrent Unit with diagonal complex state.  (Orvieto et al., 2023)
    LRU      Linear Recurrent Unit over a full sequence via associative_scan. (Orvieto et al., 2023)

Complex state parameterized via separate real/imaginary arrays.
Diagonal recurrence enables efficient per-element state updates.
Glorot normal for input/output projections, zeros for skip connection.
Sequence dim is second-to-last: (..., time, features).
Initial state defaults to zeros if not provided.
"""

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

    nu_log: Param[Float[Array, " h"]]
    theta_log: Param[Float[Array, " h"]]
    b_re: Param[Float[Array, "i h"]]
    b_im: Param[Float[Array, "i h"]]
    c_re: Param[Float[Array, "h i"]]
    c_im: Param[Float[Array, "h i"]]
    d: Param[Float[Array, " i"]]
    gamma_log: Param[Float[Array, " h"]]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        dtype: jnp.dtype = jnp.float32,
        b_init: Initializer = jax.nn.initializers.glorot_normal(),
        c_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_nu, key_theta, key_bre, key_bim, key_cre, key_cim, key_d = jax.random.split(key, 7)

        # Eigenvalue magnitudes sampled uniformly on annulus [r_min, r_max]
        u1 = jax.random.uniform(key_nu, shape=(hidden_dim,))
        u2 = jax.random.uniform(key_theta, shape=(hidden_dim,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        self.nu_log = Param(nu_log.astype(dtype))
        self.theta_log = Param(theta_log.astype(dtype))

        # Input projection B (real/imaginary parts)
        self.b_re = Param(b_init(shape=(in_dim, hidden_dim), dtype=dtype, key=key_bre))
        self.b_im = Param(b_init(shape=(in_dim, hidden_dim), dtype=dtype, key=key_bim))

        # Output projection C (real/imaginary parts)
        self.c_re = Param(c_init(shape=(hidden_dim, in_dim), dtype=dtype, key=key_cre))
        self.c_im = Param(c_init(shape=(hidden_dim, in_dim), dtype=dtype, key=key_cim))

        # Skip connection
        self.d = Param(d_init(shape=(in_dim,), dtype=dtype, key=key_d))

        # Normalization factor, derived from eigenvalue magnitudes
        self.gamma_log = Param(jnp.log(jnp.sqrt(1 - jnp.abs(self.diag_lambda) ** 2)).astype(dtype))

    @property
    def diag_lambda(self) -> Complex[Array, " h"]:
        return jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))

    @property
    def initial_state(self) -> Complex[Array, " h"]:
        return jnp.zeros(self.nu_log.shape[0], dtype=jnp.complex64)

    def __call__(
        self,
        x: Float[Array, "... i"],
        h: Complex[Array, "... h"],
    ) -> tuple[Float[Array, "... i"], Complex[Array, "... h"]]:

        b_proj = (self.b_re + 1j * self.b_im) * jnp.exp(self.gamma_log)
        c_proj = self.c_re + 1j * self.c_im

        h = self.diag_lambda * h + x.astype(jnp.complex64) @ b_proj

        return jnp.real(h @ c_proj) + self.d * x, h


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
        max_phase: float = 6.28,
        dtype: jnp.dtype = jnp.float32,
        b_init: Initializer = jax.nn.initializers.glorot_normal(),
        c_init: Initializer = jax.nn.initializers.glorot_normal(),
        d_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = LRUCell(
            in_dim, hidden_dim, r_min, r_max, max_phase, dtype, b_init, c_init, d_init, key=key
        )

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: Complex[Array, "b h"] | None = None,
    ) -> tuple[Float[Array, "b t i"], Complex[Array, "b h"]]:

        b, t, i = x.shape

        b_proj = (self.cell.b_re + 1j * self.cell.b_im) * jnp.exp(self.cell.gamma_log)
        c_proj = self.cell.c_re + 1j * self.cell.c_im

        lambdas = jnp.broadcast_to(self.cell.diag_lambda, (b, t, self.cell.nu_log.shape[0]))
        hidden = x.astype(jnp.complex64) @ b_proj

        lambdas, hidden = lax.associative_scan(fn=_binary_op, elems=(lambdas, hidden), axis=1)

        if hx is not None:
            hidden = lambdas * hx[:, None, :] + hidden

        x = jnp.real(hidden @ c_proj) + self.cell.d * x

        return x, hidden[:, -1, :]
