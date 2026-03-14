from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

from ion import nn, tree


class Linear(nn.Module):
    w: nn.Param
    b: nn.Param

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array):
        self.w = nn.Param(jax.random.normal(key, (in_dim, out_dim)))
        self.b = nn.Param(jnp.zeros(out_dim))


class WithFrozen(nn.Module):
    w: nn.Param
    frozen: nn.Param

    def __init__(self, key: jax.Array):
        self.w = nn.Param(jax.random.normal(key, (3,)))
        self.frozen = nn.Param(jnp.ones(3), trainable=False)


class MixedLeaves(nn.Module):
    w: nn.Param
    scale: int
    act: Callable

    def __init__(self, key: jax.Array):
        self.w = nn.Param(jax.random.normal(key, (2, 2)))
        self.scale = 3
        self.act = jax.nn.relu


class Nested(nn.Module):
    linear: Linear
    extra: nn.Param

    def __init__(self, key: jax.Array):
        keys = jax.random.split(key, 2)
        self.linear = Linear(2, 3, key=keys[0])
        self.extra = nn.Param(jnp.ones(3))


class TestGrad:
    def test_gradient_correctness(self):
        """Gradient of sum(w * x) w.r.t. w should be x."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w * jnp.array([4.0, 5.0, 6.0]))

        grads = jax.grad(loss)(Model())
        npt.assert_allclose(grads.w._value, jnp.array([4.0, 5.0, 6.0]))

    def test_frozen_param_gets_zero_gradient(self):
        def loss(model):
            return jnp.sum(model.w) + jnp.sum(model.frozen)

        model = WithFrozen(key=jax.random.key(0))
        grads = jax.grad(loss)(model)
        # Trainable param has a gradient
        assert jnp.any(grads.w._value != 0)
        # Frozen param gets zero gradient
        npt.assert_allclose(grads.frozen._value, jnp.zeros_like(model.frozen._value))

    def test_non_param_leaves_preserved(self):
        """Non-param leaves are preserved as static metadata in gradient output."""

        def loss(model):
            return jnp.sum(model.w) * model.scale

        model = MixedLeaves(key=jax.random.key(0))
        grads = jax.grad(loss)(model)
        assert grads.w is not None
        assert grads.scale == 3
        assert grads.act is jax.nn.relu

    def test_extra_args_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        x = jnp.array([3.0, 4.0])
        grads = jax.grad(loss)(Model(), x)
        npt.assert_allclose(grads.w._value, x)

    def test_kwargs_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, *, scale):
            return jnp.sum(model.w) * scale

        grads = jax.grad(loss)(Model(), scale=3.0)
        npt.assert_allclose(grads.w._value, jnp.array([3.0, 3.0]))

    def test_on_nested_module(self):
        def loss(model, x):
            h = x @ model.linear.w + model.linear.b
            return jnp.sum(h + model.extra)

        model = Nested(key=jax.random.key(0))
        x = jnp.ones((1, 2))
        grads = jax.grad(loss)(model, x)
        # All params are trainable → all should have gradients
        assert grads.linear.w is not None
        assert grads.linear.b is not None
        assert grads.extra is not None
        # Gradient shapes match parameter shapes
        assert grads.linear.w._value.shape == model.linear.w._value.shape
        assert grads.linear.b._value.shape == model.linear.b._value.shape
        assert grads.extra._value.shape == model.extra._value.shape


class TestGradEdgeCases:
    def test_grad_all_frozen(self):
        """jax.grad on a model where all params are frozen produces zero gradients."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]), trainable=False)

        def loss(model):
            return jnp.sum(model.w)

        grads = jax.grad(loss)(Model())
        npt.assert_allclose(grads.w._value, jnp.zeros(2))

    def test_grad_inside_jit(self):
        """jax.grad composed with jax.jit."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        x = jnp.array([3.0, 4.0])
        grads_eager = jax.grad(loss)(model, x)
        grads_jit = jax.jit(jax.grad(loss))(model, x)
        npt.assert_allclose(grads_jit.w._value, grads_eager.w._value)

    def test_grad_output_structure_matches_model(self):
        """Gradient tree has the exact same pytree structure as the model."""
        model = Nested(key=jax.random.key(0))
        x = jnp.ones((1, 2))

        def loss(m, x):
            h = x @ m.linear.w + m.linear.b
            return jnp.sum(h + m.extra)

        grads = jax.grad(loss)(model, x)
        model_leaves, model_def = jax.tree.flatten(model, is_leaf=lambda x: x is None)
        grad_leaves, grad_def = jax.tree.flatten(grads, is_leaf=lambda x: x is None)
        assert len(model_leaves) == len(grad_leaves)

    def test_grad_multiple_calls_consistent(self):
        """Calling jax.grad multiple times gives the same result."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        g1 = jax.grad(loss)(model)
        g2 = jax.grad(loss)(model)
        npt.assert_array_equal(g1.w._value, g2.w._value)

    def test_grad_partial_freeze_nested(self):
        """Gradient correctly handles partially frozen nested modules."""

        class Model(nn.Module):
            encoder: Linear
            decoder: Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.encoder = Linear(2, 3, key=keys[0])
                self.decoder = Linear(3, 1, key=keys[1])

        model = Model(key=jax.random.key(0))
        model = model.replace(encoder=model.encoder.freeze())

        def loss(model, x):
            h = x @ model.encoder.w + model.encoder.b
            return jnp.sum(h @ model.decoder.w + model.decoder.b)

        grads = jax.grad(loss)(model, jnp.ones((1, 2)))
        # Frozen encoder params get zero gradients
        npt.assert_allclose(grads.encoder.w._value, jnp.zeros_like(model.encoder.w._value))
        npt.assert_allclose(grads.encoder.b._value, jnp.zeros_like(model.encoder.b._value))
        # Trainable decoder params get gradients
        assert jnp.any(grads.decoder.w._value != 0)
        assert jnp.any(grads.decoder.b._value != 0)

    def test_value_and_grad_with_dict_pytree(self):
        """jax.value_and_grad works on dict pytrees, not just Modules."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0, 3.0]))}

        def loss(d):
            return jnp.sum(d["w"] ** 2)

        value, grads = jax.value_and_grad(loss)(data)
        npt.assert_allclose(value, 14.0)
        npt.assert_allclose(grads["w"]._value, 2.0 * jnp.asarray(data["w"]))


class TestValueAndGrad:
    def test_value_matches_direct_call(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        value, _ = jax.value_and_grad(loss)(model)
        npt.assert_allclose(value, loss(model))

    def test_grad_matches_grad_only(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        _, vag_grads = jax.value_and_grad(loss)(model)
        grad_only = jax.grad(loss)(model)
        npt.assert_allclose(vag_grads.w._value, grad_only.w._value)

    def test_frozen_param_gets_zero_gradient(self):
        def loss(model):
            return jnp.sum(model.w) + jnp.sum(model.frozen)

        model = WithFrozen(key=jax.random.key(0))
        value, grads = jax.value_and_grad(loss)(model)
        npt.assert_allclose(value, loss(model))
        assert jnp.any(grads.w._value != 0)
        npt.assert_allclose(grads.frozen._value, jnp.zeros_like(model.frozen._value))

    def test_extra_args_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        x = jnp.array([3.0, 4.0])
        value, grads = jax.value_and_grad(loss)(model, x)
        npt.assert_allclose(value, jnp.sum(model.w._value * x))
        npt.assert_allclose(grads.w._value, x)

    def test_kwargs_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, *, scale):
            return jnp.sum(model.w) * scale

        model = Model()
        value, grads = jax.value_and_grad(loss)(model, scale=3.0)
        npt.assert_allclose(value, jnp.sum(model.w._value) * 3.0)
        npt.assert_allclose(grads.w._value, jnp.array([3.0, 3.0]))

    def test_inside_jit(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        val_eager, grad_eager = jax.value_and_grad(loss)(model)
        val_jit, grad_jit = jax.jit(jax.value_and_grad(loss))(model)
        npt.assert_allclose(val_jit, val_eager)
        npt.assert_allclose(grad_jit.w._value, grad_eager.w._value)


class TestGradHasAux:
    def test_value_and_grad_has_aux(self):
        """jax.value_and_grad with has_aux=True returns ((scalar, aux), grads)."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model, x):
            out = model.w * x
            return jnp.sum(out), out

        model = Model()
        x = jnp.array([4.0, 5.0, 6.0])
        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(model, x)
        npt.assert_allclose(grads.w._value, x)
        npt.assert_allclose(aux, model.w._value * x)

    def test_value_and_grad_has_aux_frozen(self):
        """has_aux with frozen params: frozen positions get zero, aux is returned."""

        def loss(model):
            return jnp.sum(model.w) + jnp.sum(model.frozen), model.frozen._value

        model = WithFrozen(key=jax.random.key(0))
        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(model)
        assert jnp.any(grads.w._value != 0)
        npt.assert_allclose(grads.frozen._value, jnp.zeros_like(model.frozen._value))
        npt.assert_array_equal(aux, model.frozen._value)

    def test_value_and_grad_has_aux_dict(self):
        """Auxiliary output can be a dict of arrays."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model):
            s = jnp.sum(model.w**2)
            return s, {"loss": s, "w_norm": jnp.sqrt(s)}

        model = Model()
        (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(model)
        npt.assert_allclose(value, aux["loss"])
        npt.assert_allclose(aux["w_norm"], jnp.sqrt(value))


class TestGradArgnums:
    def test_argnums_non_zero(self):
        """Differentiate w.r.t. second positional argument."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(x, model):
            return jnp.sum(model.w * x)

        x = jnp.array([4.0, 5.0, 6.0])
        grads = jax.grad(loss, argnums=1)(x, Model())
        npt.assert_allclose(grads.w._value, x)

    def test_argnums_tuple(self):
        """Differentiate w.r.t. multiple arguments returns a tuple of grads."""

        class ModelA(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        class ModelB(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([3.0, 4.0]))

        def loss(a, b):
            return jnp.sum(a.w * b.w)

        ga, gb = jax.grad(loss, argnums=(0, 1))(ModelA(), ModelB())
        npt.assert_allclose(ga.w._value, jnp.array([3.0, 4.0]))
        npt.assert_allclose(gb.w._value, jnp.array([1.0, 2.0]))

    def test_value_and_grad_argnums(self):
        """value_and_grad with argnums=1."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(x, model):
            return jnp.sum(model.w * x)

        x = jnp.array([3.0, 4.0])
        value, grads = jax.value_and_grad(loss, argnums=1)(x, Model())
        npt.assert_allclose(value, 11.0)
        npt.assert_allclose(grads.w._value, x)

    def test_value_and_grad_argnums_tuple(self):
        """value_and_grad with tuple argnums returns tuple of grads."""

        class ModelA(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        class ModelB(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([3.0, 4.0]))

        def loss(a, b):
            return jnp.sum(a.w * b.w)

        value, (ga, gb) = jax.value_and_grad(loss, argnums=(0, 1))(ModelA(), ModelB())
        npt.assert_allclose(value, 11.0)
        npt.assert_allclose(ga.w._value, jnp.array([3.0, 4.0]))
        npt.assert_allclose(gb.w._value, jnp.array([1.0, 2.0]))

    def test_argnums_with_has_aux(self):
        """argnums works with has_aux=True."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(x, model):
            s = jnp.sum(model.w * x)
            return s, {"loss": s}

        x = jnp.array([3.0, 4.0])
        (value, aux), grads = jax.value_and_grad(loss, argnums=1, has_aux=True)(x, Model())
        npt.assert_allclose(grads.w._value, x)
        npt.assert_allclose(aux["loss"], 11.0)


class TestNewJaxTransforms:
    def test_jax_jacobian(self):
        """jax.jacobian on small Ion Linear, verify shape."""
        model = nn.Linear(2, 3, key=jax.random.key(0))

        def f(model, x):
            return model(x)

        x = jnp.ones(2)
        jac = jax.jacobian(f)(model, x)
        # Jacobian of output (3,) w.r.t. w (2, 3) → shape (3, 2, 3)
        assert jac.w._value.shape == (3, 2, 3)

    def test_jax_hessian(self):
        """jax.hessian on tiny scalar function of Param, verify shape."""
        p = nn.Param(jnp.array([1.0, 2.0]))

        def f(p):
            return jnp.sum(p**3)

        hess = jax.hessian(f)(p)
        # Hessian of scalar w.r.t. p (2,) → shape (2, 2)
        assert hess._value.shape == (2, 2)
        # d²/dp² of sum(p³) = 6*p on diagonal
        npt.assert_allclose(jnp.diag(hess._value), 6.0 * p._value)

    def test_jax_vmap_grad(self):
        """Per-example gradients via jax.vmap(jax.grad(fn))."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        xs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        per_example_grads = jax.vmap(jax.grad(loss), in_axes=(None, 0))(model, xs)
        # Each gradient should equal the corresponding x
        npt.assert_allclose(per_example_grads.w._value, xs)

    def test_training_loop_loss_decreases(self):
        """10-step loop: jax.value_and_grad → optax → apply_updates, assert final loss < initial loss."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(model, x, y):
            return jnp.mean((model(x) - y) ** 2)

        initial_loss = loss_fn(model, x, y)

        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = tree.apply_updates(model, updates)

        final_loss = loss_fn(model, x, y)
        assert final_loss < initial_loss

    def test_training_loop_partial_freeze(self):
        """Frozen params unchanged, loss decreases."""

        class Model(nn.Module):
            encoder: nn.Linear
            decoder: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.encoder = nn.Linear(4, 4, key=keys[0])
                self.decoder = nn.Linear(4, 1, key=keys[1])

        model = Model(key=jax.random.key(0))
        model = model.replace(encoder=model.encoder.freeze())
        frozen_w = model.encoder.w._value.copy()

        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(model, x, y):
            return jnp.mean((model.decoder(model.encoder(x)) - y) ** 2)

        initial_loss = loss_fn(model, x, y)

        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = tree.apply_updates(model, updates)

        final_loss = loss_fn(model, x, y)
        assert final_loss < initial_loss
        npt.assert_array_equal(model.encoder.w._value, frozen_w)

    def test_training_loop_batchnorm(self):
        """Running stats not corrupted by apply_updates."""
        bn = nn.BatchNorm(4, training=True)
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(bn)

        x = jax.random.normal(jax.random.key(0), (8, 4))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(bn, x)
        updates, opt_state = optimizer.update(grads, opt_state)
        result = tree.apply_updates(bn, updates)

        # Running stats (non-Param) are unchanged by apply_updates
        npt.assert_array_equal(result.running_mean, bn.running_mean)
        npt.assert_array_equal(result.running_var, bn.running_var)

    def test_frozen_conv_zero_gradient(self):
        """Frozen Conv layer produces zero grads (tests the conv.py jnp.asarray fix)."""
        conv = nn.Conv(3, 8, kernel_shape=(3, 3), padding="SAME", key=jax.random.key(0))
        conv = conv.freeze()

        x = jax.random.normal(jax.random.key(1), (1, 8, 8, 3))

        def loss_fn(conv, x):
            return jnp.sum(conv(x))

        grads = jax.grad(loss_fn)(conv, x)
        assert conv.b is not None
        assert grads.w is not None
        assert grads.b is not None
        npt.assert_allclose(grads.w._value, jnp.zeros_like(conv.w._value), atol=1e-7)
        npt.assert_allclose(grads.b._value, jnp.zeros_like(conv.b._value), atol=1e-7)

    def test_frozen_jacobian_zero(self):
        """jax.jacobian produces zeros for frozen params."""
        model = nn.Linear(2, 3, key=jax.random.key(0)).freeze()
        x = jnp.ones(2)

        jac = jax.jacobian(lambda m, x: m(x))(model, x)
        npt.assert_allclose(jac.w._value, jnp.zeros_like(jac.w._value))
        npt.assert_allclose(jac.b._value, jnp.zeros_like(jac.b._value))

    def test_frozen_hessian_zero(self):
        """jax.hessian produces zeros for frozen params."""
        p = nn.Param(jnp.array([1.0, 2.0]), trainable=False)

        hess = jax.hessian(lambda p: jnp.sum(p**3))(p)
        npt.assert_allclose(hess._value, jnp.zeros((2, 2)))

    def test_unfreeze_restores_gradient_flow(self):
        """Freezing then unfreezing restores gradient flow."""
        model = nn.Linear(2, 3, key=jax.random.key(0))
        x = jnp.ones(2)

        def loss(m, x):
            return jnp.sum(m(x))

        grads_original = jax.grad(loss)(model, x)
        grads_roundtrip = jax.grad(loss)(model.freeze().unfreeze(), x)
        npt.assert_allclose(grads_roundtrip.w._value, grads_original.w._value)
        npt.assert_allclose(grads_roundtrip.b._value, grads_original.b._value)

    def test_frozen_checkpoint_rematerialization(self):
        """stop_gradient survives jax.checkpoint (remat)."""
        model = nn.Linear(4, 1, key=jax.random.key(0)).freeze()
        x = jnp.ones(4)

        @jax.checkpoint
        def forward(model, x):
            return jnp.sum(model(x))

        grads = jax.grad(forward)(model, x)
        assert model.b is not None
        assert grads.b is not None
        npt.assert_allclose(grads.w._value, jnp.zeros_like(model.w._value))
        npt.assert_allclose(grads.b._value, jnp.zeros_like(model.b._value))

    def test_grad_of_grad_raises(self):
        """Nested jax.grad(jax.grad(f)) on Param raises because __jax_array__
        is triggered during abstractification of the intermediate gradient Param.
        Use jax.hessian instead for second-order derivatives."""
        p = nn.Param(jnp.array([1.0, 2.0]))

        def f(p):
            return jnp.sum(p**3)

        with pytest.raises(ValueError, match="__jax_array__"):
            jax.grad(jax.grad(f))(p)

    def test_grad_computes_for_non_param_arrays(self):
        """jax.grad produces gradient arrays for non-Param array leaves too."""

        class WithBuf(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (3,)))
                self.buf = jnp.ones(3)

            def __call__(self, x):
                return self.w * x + self.buf

        model = WithBuf(key=jax.random.key(0))

        def loss(m, x):
            return jnp.sum(m(x))

        grads = jax.grad(loss)(model, jnp.ones(3))
        # buf is a plain array but grad still computes a value for it
        assert grads.buf is not None
        assert isinstance(grads.buf, jax.Array)
        # apply_updates would discard this gradient (tested in test_tree.py)

    def test_optimizer_state_covers_frozen_leaves(self):
        """Optimizer allocates the same state buffers for frozen params as trainable."""
        model = Linear(2, 3, key=jax.random.key(0))
        frozen = model.freeze()

        optimizer = optax.adam(1e-3)
        state_trainable = optimizer.init(model)
        state_frozen = optimizer.init(frozen)

        # Optimizer allocates the same number of state leaves regardless of trainable flag
        leaves_t = jax.tree.leaves(state_trainable)
        leaves_f = jax.tree.leaves(state_frozen)
        assert len(leaves_t) == len(leaves_f)

        # Shapes match too, frozen params get full momentum/variance buffers
        for lt, lf in zip(leaves_t, leaves_f):
            assert lt.shape == lf.shape

    def test_partial_freeze_under_jit_grad(self):
        """Partially frozen model under jit(grad) produces correct gradients."""

        class Model(nn.Module):
            encoder: Linear
            decoder: Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.encoder = Linear(2, 3, key=keys[0])
                self.decoder = Linear(3, 1, key=keys[1])

        model = Model(key=jax.random.key(0))
        model = model.replace(encoder=model.encoder.freeze())

        def loss(model, x):
            h = x @ model.encoder.w + model.encoder.b
            return jnp.sum(h @ model.decoder.w + model.decoder.b)

        grads = jax.jit(jax.grad(loss))(model, jnp.ones((1, 2)))
        npt.assert_allclose(
            grads.encoder.w._value, jnp.zeros_like(model.encoder.w._value), atol=1e-7
        )
        assert jnp.any(grads.decoder.w._value != 0)

    def test_jit_vmap_grad(self):
        """jit(vmap(grad)) for per-example gradients."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        xs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        per_ex = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0)))(model, xs)
        npt.assert_allclose(per_ex.w._value, xs)

    def test_jit_vmap_forward(self):
        """jit(vmap(model)) forward pass composition."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (8, 4))

        eager = jax.vmap(model)(x)
        jitted = jax.jit(jax.vmap(model))(x)
        npt.assert_allclose(jitted, eager)

    def test_jax_hessian_on_module(self):
        """jax.hessian works on a Module, not just bare Param."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def f(model):
            return jnp.sum(model.w**3)

        hess = jax.hessian(f)(Model())
        # Hessian of Module returns nested structure: hess.w.w is the (2, 2) block
        npt.assert_allclose(jnp.diag(hess.w.w._value), jnp.array([6.0, 12.0]))

    def test_jitted_training_step_with_frozen_encoder(self):
        """Full JIT-compiled training step with partially frozen model."""

        class Model(nn.Module):
            encoder: nn.Linear
            decoder: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.encoder = nn.Linear(4, 4, key=keys[0])
                self.decoder = nn.Linear(4, 1, key=keys[1])

        model = Model(key=jax.random.key(0))
        model = model.replace(encoder=model.encoder.freeze())
        frozen_w = model.encoder.w._value.copy()

        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(model)

        x = jax.random.normal(jax.random.key(1), (16, 4))
        y = jnp.ones((16, 1))

        @jax.jit
        def step(model, opt_state, x, y):
            def loss_fn(m):
                return jnp.mean((m.decoder(m.encoder(x)) - y) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(model)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = tree.apply_updates(model, updates)
            return model, opt_state, loss

        initial_loss = jnp.mean((model.decoder(model.encoder(x)) - y) ** 2)
        for _ in range(20):
            model, opt_state, loss = step(model, opt_state, x, y)

        final_loss = jnp.mean((model.decoder(model.encoder(x)) - y) ** 2)
        assert final_loss < initial_loss
        npt.assert_array_equal(model.encoder.w._value, frozen_w)

    def test_lax_scan_training_loop(self):
        """Training loop via jax.lax.scan (functional iteration)."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = optax.sgd(0.01)
        opt_state = optimizer.init(model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        def scan_step(carry, _):
            model, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = tree.apply_updates(model, updates)
            return (model, opt_state), loss

        initial_loss = loss_fn(model, x, y)
        (model, _), losses = jax.lax.scan(scan_step, (model, opt_state), None, length=20)
        final_loss = loss_fn(model, x, y)

        assert final_loss < initial_loss
        assert losses[-1] < losses[0]

    def test_frozen_lora_end_to_end(self):
        """LoRA training: base weights frozen, only a/b update, loss decreases."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(4, 4, key=keys[0])
        lora = nn.LoRALinear(linear, rank=2, key=keys[1])
        frozen_base_w = lora.linear.w._value.copy()

        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(lora)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 4))

        def loss_fn(model, x, y):
            return jnp.mean((model(x) - y) ** 2)

        initial_loss = loss_fn(lora, x, y)

        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(lora, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            lora = tree.apply_updates(lora, updates)

        assert loss_fn(lora, x, y) < initial_loss
        npt.assert_array_equal(lora.linear.w._value, frozen_base_w)
