import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestLSTMCell:
    def test_output_shape(self):
        """Output (h, c) each have shape (hidden_dim,)."""
        cell = nn.LSTMCell(8, 16, key=jax.random.key(0))
        x = jnp.ones((8,))
        h, c = cell(x, cell.initial_state)
        assert h.shape == (16,)
        assert c.shape == (16,)

    def test_output_shape_batched(self):
        """Cell broadcasts over batch dimensions."""
        cell = nn.LSTMCell(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        h0 = jnp.zeros((3, 16))
        c0 = jnp.zeros((3, 16))
        h, c = cell(x, (h0, c0))
        assert h.shape == (3, 16)
        assert c.shape == (3, 16)

    def test_weight_shapes(self):
        """w_i, w_h, b have expected shapes (gate_dim = 4 * hidden_dim)."""
        cell = nn.LSTMCell(8, 16, key=jax.random.key(0))
        assert cell.w_i.shape == (8, 64)
        assert cell.w_h.shape == (16, 64)
        assert cell.b.shape == (64,)  # type: ignore[union-attr]

    def test_no_bias(self):
        """No-bias mode sets b to None."""
        cell = nn.LSTMCell(8, 16, bias=False, key=jax.random.key(0))
        assert cell.b is None

    def test_recurrent_weight_init(self):
        """Default orthogonal init on w_h: rows have unit norm (wide matrix)."""
        cell = nn.LSTMCell(16, 16, key=jax.random.key(0))
        # (16, 64) is wide, so rows are orthonormal
        row_norms = jnp.linalg.norm(cell.w_h, axis=1)
        npt.assert_allclose(row_norms, jnp.ones(16), atol=1e-5)

    def test_bias_init(self):
        """Forget gate bias is 1.0, rest are 0.0."""
        hidden_dim = 16
        cell = nn.LSTMCell(8, hidden_dim, key=jax.random.key(0))
        b = cell.b
        assert b is not None
        i_b, f_b, g_b, o_b = jnp.split(b._value, 4)
        npt.assert_array_equal(i_b, jnp.zeros(hidden_dim))
        npt.assert_array_equal(f_b, jnp.ones(hidden_dim))
        npt.assert_array_equal(g_b, jnp.zeros(hidden_dim))
        npt.assert_array_equal(o_b, jnp.zeros(hidden_dim))

    def test_initial_state_zeros(self):
        """initial_state returns zero-initialized (h, c)."""
        cell = nn.LSTMCell(8, 16, key=jax.random.key(0))
        h, c = cell.initial_state
        npt.assert_array_equal(h, jnp.zeros(16))
        npt.assert_array_equal(c, jnp.zeros(16))

    def test_manual_computation(self):
        """Output matches manual gate computation."""
        cell = nn.LSTMCell(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4,))
        h0, c0 = cell.initial_state

        h, c = cell(x, (h0, c0))

        gates = x @ cell.w_i + h0 @ cell.w_h + cell.b  # type: ignore[operator]
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_exp = f * c0 + i * g
        h_exp = o * jnp.tanh(c_exp)

        npt.assert_allclose(h, h_exp, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(c, c_exp, rtol=1e-5, atol=1e-5)


class TestGRUCell:
    def test_output_shape(self):
        """Output h has shape (hidden_dim,)."""
        cell = nn.GRUCell(8, 16, key=jax.random.key(0))
        x = jnp.ones((8,))
        h = cell(x, cell.initial_state)
        assert h.shape == (16,)

    def test_output_shape_batched(self):
        """Cell broadcasts over batch dimensions."""
        cell = nn.GRUCell(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        h0 = jnp.zeros((3, 16))
        h = cell(x, h0)
        assert h.shape == (3, 16)

    def test_weight_shapes(self):
        """w_i, w_h, b, b_h have expected shapes (gate_dim = 3 * hidden_dim)."""
        cell = nn.GRUCell(8, 16, key=jax.random.key(0))
        assert cell.w_i.shape == (8, 48)
        assert cell.w_h.shape == (16, 48)
        assert cell.b.shape == (48,)  # type: ignore[union-attr]
        assert cell.b_h.shape == (48,)  # type: ignore[union-attr]

    def test_no_bias(self):
        """No-bias mode sets b and b_h to None."""
        cell = nn.GRUCell(8, 16, bias=False, key=jax.random.key(0))
        assert cell.b is None
        assert cell.b_h is None

    def test_recurrent_weight_init(self):
        """Default orthogonal init on w_h: rows have unit norm (wide matrix)."""
        cell = nn.GRUCell(16, 16, key=jax.random.key(0))
        # (16, 48) is wide, so rows are orthonormal
        row_norms = jnp.linalg.norm(cell.w_h, axis=1)
        npt.assert_allclose(row_norms, jnp.ones(16), atol=1e-5)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        cell = nn.GRUCell(8, 16, key=jax.random.key(0))
        assert jnp.all(cell.b == 0)
        assert jnp.all(cell.b_h == 0)

    def test_initial_state_zeros(self):
        """initial_state returns zero-initialized h."""
        cell = nn.GRUCell(8, 16, key=jax.random.key(0))
        h = cell.initial_state
        npt.assert_array_equal(h, jnp.zeros(16))

    def test_manual_computation(self):
        """Output matches manual gate computation."""
        cell = nn.GRUCell(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4,))
        h0 = cell.initial_state

        h = cell(x, h0)

        gate_x = x @ cell.w_i + cell.b  # type: ignore[operator]
        gate_h = h0 @ cell.w_h + cell.b_h  # type: ignore[operator]
        r_x, z_x, n_x = jnp.split(gate_x, 3, axis=-1)
        r_h, z_h, n_h = jnp.split(gate_h, 3, axis=-1)
        r = jax.nn.sigmoid(r_x + r_h)
        z = jax.nn.sigmoid(z_x + z_h)
        n = jnp.tanh(n_x + r * n_h)
        h_exp = (1 - z) * n + z * h0

        npt.assert_allclose(h, h_exp, rtol=1e-5, atol=1e-5)


class TestLSTM:
    def test_output_shape(self):
        """Output has shape (batch, T, hidden_dim)."""
        lstm = nn.LSTM(8, 16, key=jax.random.key(0))
        x = jnp.ones((1, 5, 8))
        y, (h_n, c_n) = lstm(x)
        assert y.shape == (1, 5, 16)
        assert h_n.shape == (1, 16)
        assert c_n.shape == (1, 16)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        lstm = nn.LSTM(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 5, 8))
        y, (h_n, c_n) = lstm(x)
        assert y.shape == (3, 5, 16)
        assert h_n.shape == (3, 16)
        assert c_n.shape == (3, 16)

    def test_vmap_batch(self):
        """jax.vmap adds an extra batch dimension."""
        lstm = nn.LSTM(8, 16, key=jax.random.key(0))
        x = jnp.ones((2, 3, 5, 8))
        y, (h_n, c_n) = jax.vmap(lstm)(x)
        assert y.shape == (2, 3, 5, 16)
        assert h_n.shape == (2, 3, 16)
        assert c_n.shape == (2, 3, 16)

    def test_scan_vs_manual(self):
        """Scan-based output matches manual step-by-step unrolling."""
        lstm = nn.LSTM(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        y_scan, (h_n, c_n) = lstm(x)

        cell = lstm.cell
        h, c = cell.initial_state
        hs = []
        for t in range(3):
            h, c = cell(x[0, t], (h, c))
            hs.append(h)
        y_manual = jnp.stack(hs)

        npt.assert_allclose(y_scan[0], y_manual, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(h_n[0], h, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(c_n[0], c, rtol=1e-5, atol=1e-5)

    def test_custom_initial_state(self):
        """Custom initial state is used instead of zeros."""
        lstm = nn.LSTM(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        h0 = jnp.ones((1, 8)) * 0.5
        c0 = jnp.ones((1, 8)) * 0.1
        y_custom, _ = lstm(x, hx=(h0, c0))

        y_zero, _ = lstm(x)
        assert not jnp.allclose(y_custom, y_zero)

    def test_no_bias(self):
        """No-bias mode works through the wrapper."""
        lstm = nn.LSTM(8, 16, bias=False, key=jax.random.key(0))
        assert lstm.cell.b is None
        x = jnp.ones((1, 5, 8))
        y, (h_n, c_n) = lstm(x)
        assert y.shape == (1, 5, 16)


class TestGRU:
    def test_output_shape(self):
        """Output has shape (batch, T, hidden_dim)."""
        gru = nn.GRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((1, 5, 8))
        y, h_n = gru(x)
        assert y.shape == (1, 5, 16)
        assert h_n.shape == (1, 16)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        gru = nn.GRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 5, 8))
        y, h_n = gru(x)
        assert y.shape == (3, 5, 16)
        assert h_n.shape == (3, 16)

    def test_vmap_batch(self):
        """jax.vmap adds an extra batch dimension."""
        gru = nn.GRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((2, 3, 5, 8))
        y, h_n = jax.vmap(gru)(x)
        assert y.shape == (2, 3, 5, 16)
        assert h_n.shape == (2, 3, 16)

    def test_scan_vs_manual(self):
        """Scan-based output matches manual step-by-step unrolling."""
        gru = nn.GRU(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        y_scan, h_n = gru(x)

        cell = gru.cell
        h = cell.initial_state
        hs = []
        for t in range(3):
            h = cell(x[0, t], h)
            hs.append(h)
        y_manual = jnp.stack(hs)

        npt.assert_allclose(y_scan[0], y_manual, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(h_n[0], h, rtol=1e-5, atol=1e-5)

    def test_custom_initial_state(self):
        """Custom initial state is used instead of zeros."""
        gru = nn.GRU(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        h0 = jnp.ones((1, 8)) * 0.5
        y_custom, _ = gru(x, hx=h0)

        y_zero, _ = gru(x)
        assert not jnp.allclose(y_custom, y_zero)

    def test_no_bias(self):
        """No-bias mode works through the wrapper."""
        gru = nn.GRU(8, 16, bias=False, key=jax.random.key(0))
        assert gru.cell.b is None
        assert gru.cell.b_h is None
        x = jnp.ones((1, 5, 8))
        y, h_n = gru(x)
        assert y.shape == (1, 5, 16)
