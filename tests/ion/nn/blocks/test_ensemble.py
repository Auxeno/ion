import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestEnsemble:
    def test_matches_manual_vmap(self):
        """Default output matches construction and evaluation with vmap."""

        def factory(key):
            return nn.MLP([3, 8, 1], key=key)

        key = jax.random.key(0)
        ensemble = nn.Ensemble(factory, 4, key=key)
        x = jax.random.normal(jax.random.key(1), (5, 3))

        models = jax.vmap(factory)(jax.random.split(key, 4))
        expected = jax.vmap(lambda model: model(x))(models)

        assert isinstance(ensemble.models, nn.MLP)
        assert ensemble.models.layers[0].w.shape == (4, 3, 8)
        assert ensemble(x).shape == (4, 5, 1)
        npt.assert_allclose(ensemble(x), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction(self, reduction):
        """Configured reductions remove the leading member axis."""
        ensemble = nn.Ensemble(
            lambda key: nn.Linear(3, 2, key=key),
            4,
            reduction=reduction,
            key=jax.random.key(0),
        )
        x = jax.random.normal(jax.random.key(1), (5, 3))

        outputs = jax.vmap(lambda model: model(x))(ensemble.models)
        expected = getattr(jnp, reduction)(outputs, axis=0)

        assert ensemble(x).shape == (5, 2)
        npt.assert_allclose(ensemble(x), expected, rtol=1e-5, atol=1e-5)

    def test_routes_training_and_keys(self):
        """Each stochastic member receives an independent call key."""

        def factory(key):
            return nn.Sequential(nn.Linear(3, 3, key=key), nn.Dropout(0.5))

        ensemble = nn.Ensemble(factory, 4, key=jax.random.key(0))
        x = jnp.ones((100, 3))
        key = jax.random.key(1)

        keys = jax.random.split(key, 4)
        expected = jax.vmap(
            lambda model, model_key: model(x, training=True, key=model_key)
        )(ensemble.models, keys)

        result = ensemble(x, training=True, key=key)

        npt.assert_array_equal(result, expected)

    def test_default_reduction_is_none(self):
        """Members remain separate by default."""
        ensemble = nn.Ensemble(
            lambda key: nn.Linear(3, 2, key=key),
            4,
            key=jax.random.key(0),
        )

        assert ensemble.reduction is None
        assert ensemble(jnp.ones((5, 3))).shape == (4, 5, 2)

    def test_jit(self):
        """Ensemble evaluation works under jax.jit."""
        ensemble = nn.Ensemble(
            lambda key: nn.Linear(3, 2, key=key),
            4,
            key=jax.random.key(0),
        )
        x = jnp.ones((5, 3))

        expected = ensemble(x)
        result = jax.jit(ensemble)(x)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_size(self):
        """An ensemble must contain at least one member."""
        with pytest.raises(ValueError, match="at least 1"):
            nn.Ensemble(lambda key: nn.Linear(3, 2, key=key), 0, key=jax.random.key(0))

    def test_rejects_unknown_reduction(self):
        """Unknown reductions fail during construction."""
        with pytest.raises(ValueError, match="unknown reduction"):
            nn.Ensemble(
                lambda key: nn.Linear(3, 2, key=key),
                4,
                reduction="maximum",  # type: ignore[arg-type]
                key=jax.random.key(0),
            )

    def test_factory_must_return_module(self):
        """Factories returning other pytrees are rejected."""
        with pytest.raises(TypeError, match="factory must return a Module"):
            nn.Ensemble(
                lambda key: jax.random.normal(key, (3,)),  # type: ignore[arg-type]
                4,
                key=jax.random.key(0),
            )
