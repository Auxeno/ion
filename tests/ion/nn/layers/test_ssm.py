from math import pi

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestS4DCell:
    def test_odd_state_dim_raises(self):
        """Odd or sub-2 state_dim raises ValueError."""
        with pytest.raises(ValueError, match="even"):
            nn.S4DCell(8, 7, key=jax.random.key(0))
        with pytest.raises(ValueError, match="even"):
            nn.S4D(8, 1, key=jax.random.key(0))

    def test_output_shape(self):
        """Output y has shape (in_dim,), hx has shape (in_dim, hidden_dim//2) complex."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.shape == (8,)
        assert hx.shape == (8, 4)

    def test_output_shape_batched(self):
        """Cell broadcasts over batch dimensions."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        h0 = jnp.zeros((3, 8, 4), dtype=jnp.complex64)
        y, hx = cell(x, h0)
        assert y.shape == (3, 8)
        assert hx.shape == (3, 8, 4)

    def test_weight_shapes(self):
        """All parameters have expected shapes (hidden_dim halved for conjugate pairs)."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        assert cell.A_log_re.shape == (8, 4)
        assert cell.A_im.shape == (8, 4)
        assert cell.C.shape == (8, 4)
        assert cell.C.dtype == jnp.complex64
        assert cell.D.shape == (8,)
        assert cell.log_dt.shape == (8,)

    def test_no_b_param(self):
        """S4DCell has no B parameter (B=1, absorbed into ZOH)."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        assert not hasattr(cell, "B")

    def test_initial_state_zeros(self):
        """initial_state returns complex zeros of shape (in_dim, hidden_dim//2)."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        hx = cell.initial_state
        npt.assert_array_equal(hx, jnp.zeros((8, 4), dtype=jnp.complex64))

    def test_initial_state_dtype(self):
        """initial_state has complex64 dtype."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        assert cell.initial_state.dtype == jnp.complex64

    def test_stability(self):
        """Continuous A has strictly negative real parts (stable dynamics)."""
        cell = nn.S4DCell(8, 32, key=jax.random.key(0))
        A = -jnp.exp(cell.A_log_re._value) + 1j * cell.A_im._value
        assert jnp.all(jnp.real(A) < 0)

    def test_manual_computation(self):
        """Output matches hand-computed forward pass."""
        cell = nn.S4DCell(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4,))
        h0 = cell.initial_state

        y, hx = cell(x, h0)

        # Manual computation
        dt = jnp.exp(cell.log_dt._value)
        A = -jnp.exp(cell.A_log_re._value) + 1j * cell.A_im._value
        A_bar = jnp.exp(A * dt[:, None])
        B_bar = (A_bar - 1.0) / A

        hx_exp = A_bar * h0 + B_bar * x[:, None].astype(jnp.complex64)
        y_exp = 2.0 * jnp.real(jnp.sum(cell.C._value * hx_exp, axis=-1)) + cell.D._value * x

        npt.assert_allclose(hx, hx_exp, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(y, y_exp, rtol=1e-5, atol=1e-5)

    def test_d_init_zeros(self):
        """Skip connection D is initialized to zeros by default."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        npt.assert_array_equal(cell.D._value, jnp.zeros(8))

    def test_output_real_dtype(self):
        """Output y is real-valued (float32), hx is complex64."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.dtype == jnp.float32
        assert hx.dtype == jnp.complex64

    def test_s4d_lin_init(self):
        """S4D-Lin initialization: real parts are log(0.5), imag parts are pi*n."""
        cell = nn.S4DCell(8, 8, key=jax.random.key(0))
        expected_real = jnp.full((8, 4), jnp.log(0.5))
        expected_imag = jnp.broadcast_to(pi * jnp.arange(4), (8, 4))
        npt.assert_allclose(cell.A_log_re._value, expected_real, rtol=1e-6)
        npt.assert_allclose(cell.A_im._value, expected_imag, rtol=1e-6)


class TestS4D:
    def test_output_shape(self):
        """Outputs have shape (batch, T, in_dim), hx has shape (batch, in_dim, hidden_dim//2)."""
        s4d = nn.S4D(8, 8, key=jax.random.key(0))
        x = jnp.ones((1, 5, 8))
        outputs, hx = s4d(x)
        assert outputs.shape == (1, 5, 8)
        assert hx.shape == (1, 8, 4)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        s4d = nn.S4D(8, 8, key=jax.random.key(0))
        x = jnp.ones((3, 5, 8))
        outputs, hx = s4d(x)
        assert outputs.shape == (3, 5, 8)
        assert hx.shape == (3, 8, 4)

    def test_vmap_batch(self):
        """jax.vmap adds an extra batch dimension."""
        s4d = nn.S4D(8, 8, key=jax.random.key(0))
        x = jnp.ones((2, 3, 5, 8))
        outputs, hx = jax.vmap(s4d)(x)
        assert outputs.shape == (2, 3, 5, 8)
        assert hx.shape == (2, 3, 8, 4)

    def test_scan_vs_manual(self):
        """Scan-based output matches manual step-by-step unrolling."""
        s4d = nn.S4D(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        y_scan, h_n = s4d(x)

        cell = s4d.cell
        hx = cell.initial_state
        ys = []
        for t in range(3):
            y, hx = cell(x[0, t], hx)
            ys.append(y)
        y_manual = jnp.stack(ys)

        npt.assert_allclose(y_scan[0], y_manual, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(h_n[0], hx, rtol=1e-5, atol=1e-5)

    def test_custom_initial_state(self):
        """Custom initial state produces different output than zeros."""
        s4d = nn.S4D(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        h0 = jnp.ones((1, 4, 2), dtype=jnp.complex64) * (0.5 + 0.5j)
        y_custom, _ = s4d(x, hx=h0)

        y_zero, _ = s4d(x)
        assert not jnp.allclose(y_custom, y_zero)


class TestS5Cell:
    def test_odd_state_dim_raises(self):
        """Odd or sub-2 state_dim raises ValueError."""
        with pytest.raises(ValueError, match="even"):
            nn.S5Cell(8, 7, key=jax.random.key(0))
        with pytest.raises(ValueError, match="even"):
            nn.S5(8, 1, key=jax.random.key(0))

    def test_output_shape(self):
        """Output y has shape (in_dim,), hx has shape (hidden_dim//2,) complex."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.shape == (8,)
        assert hx.shape == (4,)

    def test_output_shape_batched(self):
        """Cell broadcasts over batch dimensions."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        h0 = jnp.zeros((3, 4), dtype=jnp.complex64)
        y, hx = cell(x, h0)
        assert y.shape == (3, 8)
        assert hx.shape == (3, 4)

    def test_weight_shapes(self):
        """All parameters have expected shapes (hidden_dim halved for conjugate pairs)."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        assert cell.A_log_re.shape == (4,)
        assert cell.A_im.shape == (4,)
        assert cell.B.shape == (8, 4)
        assert cell.B.dtype == jnp.complex64
        assert cell.C.shape == (4, 8)
        assert cell.C.dtype == jnp.complex64
        assert cell.D.shape == (8,)
        assert cell.log_dt.shape == (4,)

    def test_initial_state_zeros(self):
        """initial_state returns complex zeros of shape (hidden_dim//2,)."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        hx = cell.initial_state
        npt.assert_array_equal(hx, jnp.zeros(4, dtype=jnp.complex64))

    def test_initial_state_dtype(self):
        """initial_state has complex64 dtype."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        assert cell.initial_state.dtype == jnp.complex64

    def test_x64_complex_stays_complex64(self):
        """Under x64, B/C stay complex64 while real params become float64."""
        with jax.enable_x64(True):
            cell = nn.S5Cell(4, 8, key=jax.random.key(0))
            assert cell.B.dtype == jnp.complex64
            assert cell.A_log_re.dtype == jnp.float64
            assert cell.A_im.dtype == jnp.float64
            assert cell.log_dt.dtype == jnp.float64

    def test_stability(self):
        """Continuous A has strictly negative real parts (stable dynamics)."""
        cell = nn.S5Cell(8, 32, key=jax.random.key(0))
        A = -jnp.exp(cell.A_log_re._value) + 1j * cell.A_im._value
        assert jnp.all(jnp.real(A) < 0)

    def test_manual_computation(self):
        """Output matches hand-computed forward pass."""
        cell = nn.S5Cell(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4,))
        h0 = cell.initial_state

        y, hx = cell(x, h0)

        # Manual computation
        dt = jnp.exp(cell.log_dt._value)
        A = -jnp.exp(cell.A_log_re._value) + 1j * cell.A_im._value
        A_bar = jnp.exp(A * dt)
        B_bar = cell.B._value * ((A_bar - 1.0) / A)

        hx_exp = A_bar * h0 + x.astype(jnp.complex64) @ B_bar
        y_exp = 2.0 * jnp.real(hx_exp @ cell.C._value) + cell.D._value * x

        npt.assert_allclose(hx, hx_exp, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(y, y_exp, rtol=1e-5, atol=1e-5)

    def test_d_init_zeros(self):
        """Skip connection D is initialized to zeros by default."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        npt.assert_array_equal(cell.D._value, jnp.zeros(8))

    def test_output_real_dtype(self):
        """Output y is real-valued (float32), hx is complex64."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.dtype == jnp.float32
        assert hx.dtype == jnp.complex64

    def test_s4d_lin_init(self):
        """S4D-Lin initialization: real parts are log(0.5), imag parts are pi*n."""
        cell = nn.S5Cell(8, 8, key=jax.random.key(0))
        npt.assert_allclose(cell.A_log_re._value, jnp.full(4, jnp.log(0.5)), rtol=1e-6)
        npt.assert_allclose(cell.A_im._value, pi * jnp.arange(4), rtol=1e-6)


class TestS5:
    def test_output_shape(self):
        """Outputs have shape (batch, T, in_dim), hx has shape (batch, hidden_dim//2)."""
        s5 = nn.S5(8, 8, key=jax.random.key(0))
        x = jnp.ones((1, 5, 8))
        outputs, hx = s5(x)
        assert outputs.shape == (1, 5, 8)
        assert hx.shape == (1, 4)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        s5 = nn.S5(8, 8, key=jax.random.key(0))
        x = jnp.ones((3, 5, 8))
        outputs, hx = s5(x)
        assert outputs.shape == (3, 5, 8)
        assert hx.shape == (3, 4)

    def test_vmap_batch(self):
        """jax.vmap adds an extra batch dimension."""
        s5 = nn.S5(8, 8, key=jax.random.key(0))
        x = jnp.ones((2, 3, 5, 8))
        outputs, hx = jax.vmap(s5)(x)
        assert outputs.shape == (2, 3, 5, 8)
        assert hx.shape == (2, 3, 4)

    def test_scan_vs_manual(self):
        """Scan-based output matches manual step-by-step unrolling."""
        s5 = nn.S5(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        y_scan, h_n = s5(x)

        cell = s5.cell
        hx = cell.initial_state
        ys = []
        for t in range(3):
            y, hx = cell(x[0, t], hx)
            ys.append(y)
        y_manual = jnp.stack(ys)

        npt.assert_allclose(y_scan[0], y_manual, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(h_n[0], hx, rtol=1e-5, atol=1e-5)

    def test_custom_initial_state(self):
        """Custom initial state produces different output than zeros."""
        s5 = nn.S5(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        h0 = jnp.ones((1, 2), dtype=jnp.complex64) * (0.5 + 0.5j)
        y_custom, _ = s5(x, hx=h0)

        y_zero, _ = s5(x)
        assert not jnp.allclose(y_custom, y_zero)
