import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestBidirectional:
    def test_matches_manual_rnn(self):
        """Outputs run in opposite directions and align in original time order."""
        keys = jax.random.split(jax.random.key(0), 2)
        forward = nn.RNN(3, 4, key=keys[0])
        backward = nn.RNN(3, 4, key=keys[1])
        layer = nn.Bidirectional(forward, backward)
        x = jax.random.normal(jax.random.key(1), (2, 5, 3))

        x_forward, h_forward = forward(x)
        x_backward, h_backward = backward(jnp.flip(x, axis=1))
        expected = x_forward + jnp.flip(x_backward, axis=1)

        y, states = layer(x)

        assert y.shape == (2, 5, 4)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(states[0], h_forward, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(states[1], h_backward, rtol=1e-5, atol=1e-5)

    def test_separate_layers(self):
        """Separate forward and backward layers keep independent parameters."""
        keys = jax.random.split(jax.random.key(0), 2)
        forward = nn.RNN(3, 4, key=keys[0])
        backward = nn.RNN(3, 5, key=keys[1])
        layer = nn.Bidirectional(forward, backward, mode="concat")
        x = jax.random.normal(jax.random.key(1), (2, 6, 3))

        x_forward, _ = forward(x)
        x_backward, _ = backward(jnp.flip(x, axis=1))
        expected = jnp.concatenate((x_forward, jnp.flip(x_backward, axis=1)), axis=-1)

        y, _ = layer(x)

        assert y.shape == (2, 6, 9)
        assert layer.num_params == forward.num_params + backward.num_params
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("mode", ["sum", "mean"])
    def test_dimension_preserving_modes(self, mode):
        """Sum and mean combine equal-width directional outputs."""
        keys = jax.random.split(jax.random.key(0), 2)
        forward = nn.GRU(3, 4, key=keys[0])
        backward = nn.GRU(3, 4, key=keys[1])
        layer = nn.Bidirectional(forward, backward, mode=mode)
        x = jax.random.normal(jax.random.key(1), (2, 5, 3))

        x_forward, _ = forward(x)
        x_backward, _ = backward(jnp.flip(x, axis=1))
        expected = x_forward + jnp.flip(x_backward, axis=1)
        if mode == "mean":
            expected = expected / 2

        y, _ = layer(x)

        assert y.shape == (2, 5, 4)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_default_mode_is_sum(self):
        """The default mode preserves the directional output width."""
        keys = jax.random.split(jax.random.key(0), 2)
        layer = nn.Bidirectional(
            nn.RNN(3, 4, key=keys[0]),
            nn.RNN(3, 4, key=keys[1]),
        )

        assert layer.mode == "sum"
        assert layer(jnp.ones((2, 5, 3)))[0].shape == (2, 5, 4)

    def test_lstm_states(self):
        """Nested LSTM states are returned once for each direction."""
        keys = jax.random.split(jax.random.key(0), 2)
        layer = nn.Bidirectional(
            nn.LSTM(3, 4, key=keys[0]),
            nn.LSTM(3, 4, key=keys[1]),
        )

        y, ((h_forward, c_forward), (h_backward, c_backward)) = layer(jnp.ones((2, 5, 3)))

        assert y.shape == (2, 5, 4)
        assert h_forward.shape == c_forward.shape == (2, 4)
        assert h_backward.shape == c_backward.shape == (2, 4)

    @pytest.mark.parametrize("sequence", [nn.S4D, nn.S5])
    def test_ssm(self, sequence):
        """The block supports both associative-scan SSM layers."""
        keys = jax.random.split(jax.random.key(0), 2)
        layer = nn.Bidirectional(
            sequence(3, 8, key=keys[0]),
            sequence(3, 8, key=keys[1]),
        )

        y, states = jax.jit(layer)(jnp.ones((2, 5, 3)))

        assert y.shape == (2, 5, 3)
        assert states[0].shape[0] == states[1].shape[0] == 2

    def test_accepts_initial_states(self):
        """Initial states are routed to their corresponding directions."""
        keys = jax.random.split(jax.random.key(0), 2)
        layer = nn.Bidirectional(
            nn.RNN(3, 4, key=keys[0]),
            nn.RNN(3, 4, key=keys[1]),
        )
        x = jnp.ones((2, 5, 3))
        hx = (jnp.ones((2, 4)), -jnp.ones((2, 4)))

        y, states = layer(x, hx)

        assert y.shape == (2, 5, 4)
        assert states[0].shape == states[1].shape == (2, 4)

    def test_rejects_unknown_mode(self):
        """Unknown modes fail during construction."""
        keys = jax.random.split(jax.random.key(0), 2)
        with pytest.raises(ValueError, match="unknown mode"):
            nn.Bidirectional(
                nn.RNN(3, 4, key=keys[0]),
                nn.RNN(3, 4, key=keys[1]),
                mode="maximum",  # type: ignore[arg-type]
            )

    def test_requires_both_layers(self):
        """Both sequence directions are required explicitly."""
        with pytest.raises(TypeError):
            nn.Bidirectional(nn.RNN(3, 4, key=jax.random.key(0)))  # type: ignore[call-arg]
