"""Recurrent layers and cells.

Modules:
    RNNCell   Single-step vanilla RNN with tanh activation.                 (Elman, 1990)
    RNN       Vanilla RNN over a full sequence via lax.scan.                (Elman, 1990)
    LSTMCell  Single-step LSTM with input, forget, cell, and output gates.  (Hochreiter & Schmidhuber, 1997)
    LSTM      LSTM over a full sequence via lax.scan.                       (Hochreiter & Schmidhuber, 1997)
    GRUCell   Single-step GRU with reset, update, and new gates.            (Cho et al., 2014)
    GRU       GRU over a full sequence via lax.scan.                        (Cho et al., 2014)

Sequence layers use sequential scan for O(T) parallel time complexity.
Glorot uniform for input weights, orthogonal for hidden weights.
LSTMCell forget gate bias initialized to 1 to encourage remembering.
Input layout is (batch, time, features).
Initial state defaults to zeros if not provided.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class RNNCell(Module):
    """Single-step vanilla RNN cell.

    >>> cell = RNNCell(3, 16, key=key)
    >>> h = cell(x, h)  # (*, 3), (*, 16) -> (*, 16)
    """

    w_i: Param[Float[Array, "i h"]]
    w_h: Param[Float[Array, "h h"]]
    b: Param[Float[Array, " h"]] | None

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
        self.w_i = Param(w_i_init(shape=(in_dim, hidden_dim), dtype=dtype, key=key_wi))
        self.w_h = Param(w_h_init(shape=(hidden_dim, hidden_dim), dtype=dtype, key=key_wh))
        self.b = Param(b_init(shape=(hidden_dim,), dtype=dtype, key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "... i"],
        h: Float[Array, "... h"],
    ) -> Float[Array, "... h"]:

        h = x @ self.w_i + h @ self.w_h
        if self.b is not None:
            h = h + self.b

        return jnp.tanh(h)

    @property
    def initial_state(self) -> Float[Array, " h"]:
        return jnp.zeros(self.w_h.shape[0])


class RNN(Module):
    """Vanilla RNN over a full sequence.

    >>> rnn = RNN(3, 16, key=key)
    >>> outputs, h = rnn(x)  # (b, t, 3) -> (b, t, 16), (b, 16)
    """

    cell: RNNCell

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

        self.cell = RNNCell(in_dim, hidden_dim, bias, dtype, w_i_init, w_h_init, b_init, key=key)

    def __call__(
        self,
        x: Float[Array, "b t i"],
        hx: Float[Array, "b h"] | None = None,
    ) -> tuple[Float[Array, "b t h"], Float[Array, "b h"]]:

        b, t, i = x.shape
        hd = self.cell.w_h.shape[0]

        if hx is None:
            h0 = self.cell.initial_state
            h0 = jnp.broadcast_to(h0, (b, hd))
        else:
            h0 = hx

        def step(carry, x_t):
            h = self.cell(x_t, carry)
            return h, h

        hx, x = lax.scan(f=step, init=h0, xs=jnp.moveaxis(x, 1, 0))
        x = jnp.moveaxis(x, 0, 1)

        return x, hx


class LSTMCell(Module):
    """Single-step LSTM cell.

    >>> cell = LSTMCell(3, 16, key=key)
    >>> h, c = cell(x, (h, c))  # (*, 3), ((*, 16), (*, 16)) -> ((*, 16), (*, 16))
    """

    w_i: Param[Float[Array, "i g"]]
    w_h: Param[Float[Array, "h g"]]
    b: Param[Float[Array, " g"]] | None

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
        x: Float[Array, "... i"],
        hx: tuple[Float[Array, "... h"], Float[Array, "... h"]],
    ) -> tuple[Float[Array, "... h"], Float[Array, "... h"]]:

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
    def initial_state(self) -> tuple[Float[Array, " h"], Float[Array, " h"]]:
        hd = self.w_h.shape[0]
        return (jnp.zeros(hd), jnp.zeros(hd))


class LSTM(Module):
    """LSTM over a full sequence.

    >>> lstm = LSTM(3, 16, key=key)
    >>> outputs, (h, c) = lstm(x)  # (b, t, 3) -> (b, t, 16), ((b, 16), (b, 16))
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
        x: Float[Array, "b t i"],
        hx: tuple[Float[Array, "b h"], Float[Array, "b h"]] | None = None,
    ) -> tuple[Float[Array, "b t h"], tuple[Float[Array, "b h"], Float[Array, "b h"]]]:

        b, t, i = x.shape
        hd = self.cell.w_h.shape[0]

        if hx is None:
            h0, c0 = self.cell.initial_state
            h0 = jnp.broadcast_to(h0, (b, hd))
            c0 = jnp.broadcast_to(c0, (b, hd))
        else:
            h0, c0 = hx

        def step(carry, x_t):
            out = self.cell(x_t, carry)
            return out, out[0]

        hx, x = lax.scan(f=step, init=(h0, c0), xs=jnp.moveaxis(x, 1, 0))
        x = jnp.moveaxis(x, 0, 1)

        return x, hx


class GRUCell(Module):
    """Single-step GRU cell.

    >>> cell = GRUCell(3, 16, key=key)
    >>> h = cell(x, h)  # (*, 3), (*, 16) -> (*, 16)
    """

    w_i: Param[Float[Array, "i g"]]
    w_h: Param[Float[Array, "h g"]]
    b: Param[Float[Array, " g"]] | None
    b_h: Param[Float[Array, " g"]] | None

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
        x: Float[Array, "... i"],
        h: Float[Array, "... h"],
    ) -> Float[Array, "... h"]:

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
    def initial_state(self) -> Float[Array, " h"]:
        return jnp.zeros(self.w_h.shape[0])


class GRU(Module):
    """GRU over a full sequence.

    >>> gru = GRU(3, 16, key=key)
    >>> outputs, h = gru(x)  # (b, t, 3) -> (b, t, 16), (b, 16)
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
        x: Float[Array, "b t i"],
        hx: Float[Array, "b h"] | None = None,
    ) -> tuple[Float[Array, "b t h"], Float[Array, "b h"]]:

        b, t, i = x.shape
        hd = self.cell.w_h.shape[0]

        if hx is None:
            h0 = self.cell.initial_state
            h0 = jnp.broadcast_to(h0, (b, hd))
        else:
            h0 = hx

        def step(carry, x_t):
            h = self.cell(x_t, carry)
            return h, h

        hx, x = lax.scan(f=step, init=h0, xs=jnp.moveaxis(x, 1, 0))
        x = jnp.moveaxis(x, 0, 1)

        return x, hx
