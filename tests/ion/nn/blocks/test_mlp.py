import jax
import jax.numpy as jnp

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

