import jax
import jax.numpy as jnp
import pytest

from ion import nn


class TestMLP:
    def test_two_dims_single_layer(self):
        """dims=[in, out] creates 1 Linear layer (in->out)."""
        mlp = nn.MLP([8, 16], key=jax.random.key(0))
        assert len(mlp.layers) == 1  # in->out

    def test_one_hidden(self):
        """dims=[in, hidden, out] creates 2 Linear layers (in->hidden, hidden->out)."""
        mlp = nn.MLP([8, 32, 16], key=jax.random.key(0))
        assert len(mlp.layers) == 2  # in->hidden, hidden->out

    def test_three_hidden(self):
        """dims with 3 hidden widths creates 4 Linear layers."""
        mlp = nn.MLP([8, 32, 32, 32, 16], key=jax.random.key(0))
        assert len(mlp.layers) == 4  # in->hidden, hidden->hidden, hidden->hidden, hidden->out

    def test_output_shape_different_dims(self):
        """Output shape matches the last dim when input and output dims differ."""
        mlp = nn.MLP([8, 32, 32, 16], key=jax.random.key(0))
        x = jnp.ones((8,))
        y = mlp(x)
        assert y.shape == (16,)

    def test_varying_widths(self):
        """Consecutive dims pairs set each Linear layer's weight shape."""
        mlp = nn.MLP([8, 32, 16, 4], key=jax.random.key(0))
        assert [layer.w.shape for layer in mlp.layers] == [(8, 32), (32, 16), (16, 4)]

    def test_single_dim_raises(self):
        """dims with fewer than 2 entries raises ValueError."""
        with pytest.raises(ValueError, match="at least an input and output dim"):
            nn.MLP([8], key=jax.random.key(0))

    def test_final_activation(self):
        """final_activation is applied to the output."""
        mlp = nn.MLP([3, 8, 1], final_activation=jax.nn.sigmoid, key=jax.random.key(0))
        x = jnp.ones((2, 3))
        y = mlp(x)
        assert jnp.all((y >= 0) & (y <= 1))
