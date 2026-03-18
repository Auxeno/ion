import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestLRUCell:
    def test_output_shape(self):
        """Output y has shape (in_dim,), hx has shape (state_dim,) complex."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.shape == (8,)
        assert hx.shape == (16,)

    def test_output_shape_batched(self):
        """Cell broadcasts over batch dimensions."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        h0 = jnp.zeros((3, 16), dtype=jnp.complex64)
        y, hx = cell(x, h0)
        assert y.shape == (3, 8)
        assert hx.shape == (3, 16)

    def test_weight_shapes(self):
        """All parameters have expected shapes."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        assert cell.nu_log.shape == (16,)
        assert cell.theta_log.shape == (16,)
        assert cell.b_re.shape == (8, 16)
        assert cell.b_im.shape == (8, 16)
        assert cell.c_re.shape == (16, 8)
        assert cell.c_im.shape == (16, 8)
        assert cell.d.shape == (8,)
        assert cell.gamma_log.shape == (16,)

    def test_initial_state_zeros(self):
        """initial_state returns complex zeros of shape (state_dim,)."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        hx = cell.initial_state
        npt.assert_array_equal(hx, jnp.zeros(16, dtype=jnp.complex64))

    def test_initial_state_dtype(self):
        """initial_state has complex64 dtype."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        assert cell.initial_state.dtype == jnp.complex64

    def test_eigenvalue_magnitude_bounds(self):
        """Eigenvalue magnitudes lie in [r_min, r_max] after initialization."""
        r_min, r_max = 0.5, 0.9
        cell = nn.LRUCell(8, 64, r_min=r_min, r_max=r_max, key=jax.random.key(0))
        diag_lambda = jnp.exp(-jnp.exp(cell.nu_log._value) + 1j * jnp.exp(cell.theta_log._value))
        magnitudes = jnp.abs(diag_lambda)
        assert jnp.all(magnitudes >= r_min - 1e-5)
        assert jnp.all(magnitudes <= r_max + 1e-5)

    def test_manual_computation(self):
        """Output matches hand-computed forward pass."""
        cell = nn.LRUCell(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4,))
        h0 = cell.initial_state

        y, hx = cell(x, h0)

        # Manual computation
        lam = jnp.exp(-jnp.exp(cell.nu_log._value) + 1j * jnp.exp(cell.theta_log._value))
        b = (cell.b_re._value + 1j * cell.b_im._value) * jnp.exp(cell.gamma_log._value)
        c = cell.c_re._value + 1j * cell.c_im._value

        hx_exp = lam * h0 + x.astype(jnp.complex64) @ b
        y_exp = jnp.real(hx_exp @ c) + cell.d._value * x

        npt.assert_allclose(hx, hx_exp, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(y, y_exp, rtol=1e-5, atol=1e-5)

    def test_d_init_zeros(self):
        """Skip connection d is initialized to zeros by default."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        npt.assert_array_equal(cell.d._value, jnp.zeros(8))

    def test_output_real_dtype(self):
        """Output y is real-valued (float32), hx is complex64."""
        cell = nn.LRUCell(8, 16, key=jax.random.key(0))
        y, hx = cell(jnp.ones((8,)), cell.initial_state)
        assert y.dtype == jnp.float32
        assert hx.dtype == jnp.complex64


class TestLRU:
    def test_output_shape(self):
        """Outputs have shape (batch, T, in_dim), hx has shape (batch, state_dim)."""
        lru = nn.LRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((1, 5, 8))
        outputs, hx = lru(x)
        assert outputs.shape == (1, 5, 8)
        assert hx.shape == (1, 16)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        lru = nn.LRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 5, 8))
        outputs, hx = lru(x)
        assert outputs.shape == (3, 5, 8)
        assert hx.shape == (3, 16)

    def test_vmap_batch(self):
        """jax.vmap adds an extra batch dimension."""
        lru = nn.LRU(8, 16, key=jax.random.key(0))
        x = jnp.ones((2, 3, 5, 8))
        outputs, hx = jax.vmap(lru)(x)
        assert outputs.shape == (2, 3, 5, 8)
        assert hx.shape == (2, 3, 16)

    def test_scan_vs_manual(self):
        """Scan-based output matches manual step-by-step unrolling."""
        lru = nn.LRU(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        y_scan, h_n = lru(x)

        cell = lru.cell
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
        lru = nn.LRU(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 4))

        h0 = jnp.ones((1, 8), dtype=jnp.complex64) * (0.5 + 0.5j)
        y_custom, _ = lru(x, hx=h0)

        y_zero, _ = lru(x)
        assert not jnp.allclose(y_custom, y_zero)
