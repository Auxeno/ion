"""Recurrent layers and cells.

Modules:
    LSTMCell  Single-step LSTM with input, forget, cell, and output gates.  (Hochreiter & Schmidhuber, 1997)
    GRUCell   Single-step GRU with reset, update, and new gates.            (Cho et al., 2014)
    LSTM      LSTM over a full sequence via lax.scan.                       (Hochreiter & Schmidhuber, 1997)
    GRU       GRU over a full sequence via lax.scan.                        (Cho et al., 2014)

Glorot uniform for input weights, orthogonal for hidden weights.
LSTMCell forget gate bias initialized to 1 to encourage remembering.
Sequence dim is second-to-last: (..., time, features).
Initial state defaults to zeros if not provided.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class LSTMCell(Module):
    """Single-step LSTM cell.

    >>> cell = LSTMCell(3, 16, key=key)
    >>> h, c = cell(x, (h, c))  # (*, 3), ((*, 16), (*, 16)) -> ((*, 16), (*, 16))
    """

    w_i: Param[Float[Array, "id gd"]]
    w_h: Param[Float[Array, "hd gd"]]
    b: Param[Float[Array, " gd"]] | None

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_i_init: Initializer = jax.nn.initializers.glorot_uniform(),
        w_h_init: Initializer = jax.nn.initializers.orthogonal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_wi, key_wh, key_b = jax.random.split(key, 3)
        gate_dim = 4 * hidden_dim
        self.w_i = Param(w_i_init(shape=(in_dim, gate_dim), dtype=dtype, key=key_wi))
        self.w_h = Param(w_h_init(shape=(hidden_dim, gate_dim), dtype=dtype, key=key_wh))
        if bias:
            b = b_init(shape=(gate_dim,), dtype=dtype, key=key_b)

            # Initialise forget gate weights to 1.0
            i, f, g, o = jnp.split(b, 4)
            self.b = Param(jnp.concatenate((i, jnp.ones_like(f), g, o)))
        else:
            self.b = None

    def __call__(
        self,
        x: Float[Array, "... id"],
        hx: tuple[Float[Array, "... hd"], Float[Array, "... hd"]],
    ) -> tuple[Float[Array, "... hd"], Float[Array, "... hd"]]:

        h, c = hx

        gates = x @ self.w_i + h @ self.w_h
        if self.b is not None:
            gates = gates + self.b

        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c = f * c + i * g
        h = o * jnp.tanh(c)

        return (h, c)

    @property
    def initial_state(self) -> tuple[Float[Array, " hd"], Float[Array, " hd"]]:
        hd = self.w_h.shape[0]
        return (jnp.zeros(hd), jnp.zeros(hd))


class GRUCell(Module):
    """Single-step GRU cell.

    >>> cell = GRUCell(3, 16, key=key)
    >>> h = cell(x, h)  # (*, 3), (*, 16) -> (*, 16)
    """

    w_i: Param[Float[Array, "id gd"]]
    w_h: Param[Float[Array, "hd gd"]]
    b: Param[Float[Array, " gd"]] | None
    b_h: Param[Float[Array, " gd"]] | None

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_i_init: Initializer = jax.nn.initializers.glorot_uniform(),
        w_h_init: Initializer = jax.nn.initializers.orthogonal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_wi, key_wh, key_b, key_bh = jax.random.split(key, 4)
        gate_dim = 3 * hidden_dim
        self.w_i = Param(w_i_init(shape=(in_dim, gate_dim), dtype=dtype, key=key_wi))
        self.w_h = Param(w_h_init(shape=(hidden_dim, gate_dim), dtype=dtype, key=key_wh))
        self.b = Param(b_init(shape=(gate_dim,), dtype=dtype, key=key_b)) if bias else None
        self.b_h = Param(b_init(shape=(gate_dim,), dtype=dtype, key=key_bh)) if bias else None

    def __call__(
        self,
        x: Float[Array, "... id"],
        h: Float[Array, "... hd"],
    ) -> Float[Array, "... hd"]:

        gate_x = x @ self.w_i
        gate_h = h @ self.w_h

        if self.b is not None:
            gate_x = gate_x + self.b
        if self.b_h is not None:
            gate_h = gate_h + self.b_h

        r_x, z_x, n_x = jnp.split(gate_x, 3, axis=-1)
        r_h, z_h, n_h = jnp.split(gate_h, 3, axis=-1)

        r = jax.nn.sigmoid(r_x + r_h)
        z = jax.nn.sigmoid(z_x + z_h)
        n = jnp.tanh(n_x + r * n_h)

        h = (1 - z) * n + z * h

        return h

    @property
    def initial_state(self) -> Float[Array, " hd"]:
        return jnp.zeros(self.w_h.shape[0])


class LSTM(Module):
    """LSTM over a full sequence.

    >>> lstm = LSTM(3, 16, key=key)
    >>> outputs, (h, c) = lstm(x)  # (*, t, 3) -> (*, t, 16), ((*, 16), (*, 16))
    """

    cell: LSTMCell

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_i_init: Initializer = jax.nn.initializers.glorot_uniform(),
        w_h_init: Initializer = jax.nn.initializers.orthogonal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = LSTMCell(in_dim, hidden_dim, bias, dtype, w_i_init, w_h_init, b_init, key=key)

    def __call__(
        self,
        x: Float[Array, "... t id"],
        hx: tuple[Float[Array, "... hd"], Float[Array, "... hd"]] | None = None,
    ) -> tuple[Float[Array, "... t hd"], tuple[Float[Array, "... hd"], Float[Array, "... hd"]]]:

        batch_shape = x.shape[:-2]
        t = x.shape[-2]
        hd = self.cell.w_h.shape[0]

        x_flat = x.reshape(-1, t, x.shape[-1])
        b = x_flat.shape[0]

        if hx is None:
            h0, c0 = self.cell.initial_state
            h0 = jnp.broadcast_to(h0, (b, hd))
            c0 = jnp.broadcast_to(c0, (b, hd))
        else:
            h0 = hx[0].reshape(-1, hd)
            c0 = hx[1].reshape(-1, hd)

        def step(carry, x_t):
            out = self.cell(x_t, carry)
            return out, out[0]

        x_time = jnp.moveaxis(x_flat, -2, 0)
        hx, x = lax.scan(f=step, init=(h0, c0), xs=x_time)

        x = jnp.moveaxis(x, 0, -2)
        x = x.reshape(*batch_shape, t, hd)
        hx = (hx[0].reshape(*batch_shape, hd), hx[1].reshape(*batch_shape, hd))

        return x, hx


class GRU(Module):
    """GRU over a full sequence.

    >>> gru = GRU(3, 16, key=key)
    >>> outputs, h = gru(x)  # (*, t, 3) -> (*, t, 16), (*, 16)
    """

    cell: GRUCell

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_i_init: Initializer = jax.nn.initializers.glorot_uniform(),
        w_h_init: Initializer = jax.nn.initializers.orthogonal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.cell = GRUCell(in_dim, hidden_dim, bias, dtype, w_i_init, w_h_init, b_init, key=key)

    def __call__(
        self,
        x: Float[Array, "... t id"],
        hx: Float[Array, "... hd"] | None = None,
    ) -> tuple[Float[Array, "... t hd"], Float[Array, "... hd"]]:

        batch_shape = x.shape[:-2]
        t = x.shape[-2]
        hd = self.cell.w_h.shape[0]

        x_flat = x.reshape(-1, t, x.shape[-1])
        b = x_flat.shape[0]

        if hx is None:
            h0 = self.cell.initial_state
            h0 = jnp.broadcast_to(h0, (b, hd))
        else:
            h0 = hx.reshape(-1, hd)

        def step(carry, x_t):
            h = self.cell(x_t, carry)
            return h, h

        x_time = jnp.moveaxis(x_flat, -2, 0)
        hx, x = lax.scan(f=step, init=h0, xs=x_time)

        x = jnp.moveaxis(x, 0, -2)
        x = x.reshape(*batch_shape, t, hd)
        hx = hx.reshape(*batch_shape, hd)

        return x, hx
