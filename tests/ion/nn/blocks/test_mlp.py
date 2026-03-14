import jax
import jax.numpy as jnp
import pytest

from ion import nn


class TestMLP:
    def test_num_layers_zero_hidden(self):
        """num_hidden_layers=0 creates 1 Linear layer (in->out)."""
        mlp = nn.MLP(8, 16, 32, num_hidden_layers=0, key=jax.random.key(0))
        assert len(mlp.layers) == 1  # in->out

    def test_num_layers_one_hidden(self):
        """num_hidden_layers=1 creates 2 Linear layers (in->hidden, hidden->out)."""
        mlp = nn.MLP(8, 16, 32, num_hidden_layers=1, key=jax.random.key(0))
        assert len(mlp.layers) == 2  # in->hidden, hidden->out

    def test_num_layers_three_hidden(self):
        """num_hidden_layers=3 creates 4 Linear layers (in + 2 hidden + out)."""
        mlp = nn.MLP(8, 16, 32, num_hidden_layers=3, key=jax.random.key(0))
        assert len(mlp.layers) == 4  # in->hidden, hidden->hidden, hidden->hidden, hidden->out

    def test_output_shape_different_dims(self):
        """Output shape matches out_dim when in_dim != out_dim."""
        mlp = nn.MLP(8, 16, 32, num_hidden_layers=2, key=jax.random.key(0))
        x = jnp.ones((8,))
        y = mlp(x)
        assert y.shape == (16,)

    def test_negative_hidden_layers_raises(self):
        """Negative num_hidden_layers raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            nn.MLP(8, 16, 32, num_hidden_layers=-1, key=jax.random.key(0))

    def test_final_activation(self):
        """final_activation is applied to the output."""
        mlp = nn.MLP(
            3,
            1,
            hidden_dim=8,
            num_hidden_layers=1,
            final_activation=jax.nn.sigmoid,
            key=jax.random.key(0),
        )
        x = jnp.ones((2, 3))
        y = mlp(x)
        assert jnp.all((y >= 0) & (y <= 1))
