import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestEmbedding:
    def test_output_manual(self):
        """Output matches direct weight table lookup."""
        key = jax.random.key(0)
        emb = nn.Embedding(10, 8, key=key)
        x = jnp.array([0, 3, 9])
        y = emb(x)
        expected = emb.w[x]
        npt.assert_allclose(y, expected, rtol=0, atol=0)

    def test_output_shape(self):
        """Output shape is (*input_shape, dim)."""
        key = jax.random.key(0)
        emb = nn.Embedding(16, 32, key=key)
        x = jnp.array([1, 2, 3])
        assert emb(x).shape == (3, 32)

    def test_scalar_index(self):
        """A scalar index returns a single embedding vector."""
        key = jax.random.key(0)
        emb = nn.Embedding(10, 8, key=key)
        x = jnp.array(5)
        assert emb(x).shape == (8,)

    def test_2d_indices(self):
        """2D index array returns embeddings with matching leading dims."""
        key = jax.random.key(0)
        emb = nn.Embedding(10, 8, key=key)
        x = jnp.array([[0, 1], [2, 3]])
        assert emb(x).shape == (2, 2, 8)

    def test_weight_shape(self):
        """Weight matrix has shape (num_embeddings, dim)."""
        key = jax.random.key(0)
        emb = nn.Embedding(20, 64, key=key)
        assert emb.w.shape == (20, 64)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        key = jax.random.key(0)
        emb = nn.Embedding(10, 8, dtype=jnp.float32, key=key)
        assert emb.w.dtype == jnp.float32

    def test_different_keys(self):
        """Different PRNG keys produce different weights."""
        e1 = nn.Embedding(10, 8, key=jax.random.key(0))
        e2 = nn.Embedding(10, 8, key=jax.random.key(1))
        assert not jnp.array_equal(e1.w.value, e2.w.value)
