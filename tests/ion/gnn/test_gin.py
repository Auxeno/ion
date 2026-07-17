import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion.gnn import GINConv
from ion.nn import MLP
from ion.nn.param import Param


class TestGINConv:
    def test_output_manual(self):
        """Output matches manual MLP((1 + eps) * x + sum-aggregation)."""
        mlp = MLP([4, 8, 8], key=jax.random.key(0))
        gin = GINConv(mlp, eps=0.5)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        y = gin(x, senders, receivers)
        agg = jax.ops.segment_sum(x[senders], receivers, 3)
        expected = mlp(1.5 * x + agg)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        """Output shape is (n, out_dim) of the update MLP."""
        gin = GINConv(MLP([4, 8, 8], key=jax.random.key(0)))
        x = jax.random.normal(jax.random.key(1), (5, 4))
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        assert gin(x, senders, receivers).shape == (5, 8)

    def test_default_eps_static(self):
        """Default eps is a plain float, not a trainable Param."""
        mlp = MLP([4, 8], key=jax.random.key(0))
        gin = GINConv(mlp)
        assert isinstance(gin.eps, float)
        assert gin.num_params == mlp.num_params

    def test_train_eps(self):
        """train_eps=True makes eps a scalar Param that receives gradient."""
        mlp = MLP([4, 8], key=jax.random.key(0))
        gin = GINConv(mlp, train_eps=True)
        assert isinstance(gin.eps, Param)
        assert gin.num_params == mlp.num_params + 1
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        grads = jax.grad(lambda m: m(x, senders, receivers).sum())(gin)
        assert jnp.any(grads.eps._value != 0)

    def test_isolated_node(self):
        """A node with no incoming edges gets MLP((1 + eps) * x) only."""
        mlp = MLP([4, 8], key=jax.random.key(0))
        gin = GINConv(mlp, eps=0.25)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        y = gin(x, senders, receivers)
        npt.assert_allclose(y[2], mlp(1.25 * x[2]), rtol=1e-5, atol=1e-5)

    def test_sum_aggregation_counts(self):
        """Duplicate neighbor features accumulate (sum, not mean)."""
        mlp = MLP([2, 4], key=jax.random.key(0))
        gin = GINConv(mlp)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        # Node 2 receives from node 0 once vs from nodes 0 and 1 (same feature)
        y_once = gin(x, jnp.array([0]), jnp.array([2]))
        y_twice = gin(x, jnp.array([0, 1]), jnp.array([2, 2]))
        assert not jnp.allclose(y_once[2], y_twice[2])

    def test_default_dtype(self):
        """Trainable eps defaults to float32."""
        gin = GINConv(MLP([4, 8], key=jax.random.key(0)), train_eps=True)
        assert isinstance(gin.eps, Param)
        assert gin.eps.dtype == jnp.float32
