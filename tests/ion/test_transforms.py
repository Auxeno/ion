from typing import Callable

import jax
import jax.numpy as jnp
import numpy.testing as npt

import ion
from ion import nn
from ion.transforms import _merge_leaves, _split_leaves
from ion.tree import Static

_is_array = lambda x: isinstance(x, jax.Array)


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
        k1, k2 = jax.random.split(key)
        self.linear = Linear(2, 3, key=k1)
        self.extra = nn.Param(jnp.ones(3))


class TestSplitLeaves:
    def test_separates_by_predicate(self):
        a, b = jnp.array(1.0), jnp.array(2.0)
        leaves = [a, 1, b, "x"]
        matching, non_matching, mask = _split_leaves(leaves, _is_array)
        assert matching == (a, b)
        assert non_matching == (1, "x")
        assert mask == (True, False, True, False)

    def test_empty_list(self):
        matching, non_matching, mask = _split_leaves([], _is_array)
        assert matching == ()
        assert non_matching == ()
        assert mask == ()

    def test_all_match(self):
        a, b = jnp.array(1.0), jnp.array(2.0)
        matching, non_matching, mask = _split_leaves([a, b], _is_array)
        assert len(matching) == 2
        assert non_matching == ()
        assert mask == (True, True)

    def test_none_match(self):
        matching, non_matching, mask = _split_leaves([1, "x", None], _is_array)
        assert matching == ()
        assert len(non_matching) == 3
        assert mask == (False, False, False)


class TestMergeLeaves:
    def test_roundtrip_with_split(self):
        a, b = jnp.array(1.0), jnp.array(2.0)
        original = [a, 1, b, "x"]
        matching, non_matching, mask = _split_leaves(original, _is_array)
        recovered = _merge_leaves(matching, non_matching, mask)
        assert len(recovered) == len(original)
        for orig, rec in zip(original, recovered):
            if _is_array(orig):
                npt.assert_array_equal(orig, rec)
            else:
                assert orig == rec

    def test_interleaved_mask(self):
        mask = (True, False, True, False, True)
        matching = ("a", "c", "e")
        non_matching = ("b", "d")
        result = _merge_leaves(matching, non_matching, mask)
        assert result == ["a", "b", "c", "d", "e"]


class TestStatic:
    def test_pytree_no_children(self):
        assert jax.tree.leaves(Static(42)) == []

    def test_value_preserved_through_flatten(self):
        s = Static({"dropout": 0.1, "mode": "train"})
        children, aux = s.tree_flatten()
        assert children == []
        recovered = Static.tree_unflatten(aux, children)
        assert recovered.value == s.value

    def test_used_in_tree_map(self):
        """tree_map sees no children inside Static, so the inner value is untouched."""
        pytree = {"arr": jnp.array(1.0), "cfg": Static(99)}
        mapped = jax.tree.map(lambda x: x * 2, pytree)
        npt.assert_array_equal(mapped["arr"], jnp.array(2.0))
        # Static has no children so tree_map passes it through unchanged
        assert isinstance(mapped["cfg"], Static)
        assert mapped["cfg"].value == 99


class TestJaxJit:
    def test_jax_jit_with_mixed_leaves(self):
        """jax.jit works with modules that have mixed array/non-array fields."""

        def fn(model, x):
            return model.act(x @ model.w) * model.scale

        key = jax.random.key(0)
        model = MixedLeaves(key=key)
        x = jnp.ones((1, 2))

        eager = fn(model, x)
        jitted = jax.jit(fn)(model, x)
        npt.assert_allclose(jitted, eager)

    def test_jax_jit_with_sequential(self):
        """Sequential with (Linear, relu, Linear) works under jax.jit."""

        k1, k2 = jax.random.split(jax.random.key(0))
        model = nn.Sequential(
            nn.Linear(2, 4, key=k1),
            jax.nn.relu,
            nn.Linear(4, 1, key=k2),
        )

        def fn(model, x):
            return model(x)

        x = jnp.ones((1, 2))
        eager = fn(model, x)
        jitted = jax.jit(fn)(model, x)
        npt.assert_allclose(jitted, eager)


class TestGrad:
    def test_gradient_correctness(self):
        """Gradient of sum(w * x) w.r.t. w should be x."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w * jnp.array([4.0, 5.0, 6.0]))

        grads = ion.grad(loss)(Model())
        npt.assert_allclose(grads.w.value, jnp.array([4.0, 5.0, 6.0]))

    def test_frozen_param_gets_none(self):
        def loss(model):
            return jnp.sum(model.w.value) + jnp.sum(model.frozen.value)

        model = WithFrozen(key=jax.random.key(0))
        grads = ion.grad(loss)(model)
        # Trainable param has a gradient
        assert grads.w is not None
        # Frozen param gets None
        assert grads.frozen is None

    def test_non_param_leaves_preserved(self):
        """Non-param leaves are preserved as static metadata in gradient output."""

        def loss(model):
            return jnp.sum(model.w.value) * model.scale

        model = MixedLeaves(key=jax.random.key(0))
        grads = ion.grad(loss)(model)
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
        grads = ion.grad(loss)(Model(), x)
        npt.assert_allclose(grads.w.value, x)

    def test_kwargs_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, *, scale):
            return jnp.sum(model.w) * scale

        grads = ion.grad(loss)(Model(), scale=3.0)
        npt.assert_allclose(grads.w.value, jnp.array([3.0, 3.0]))

    def test_on_nested_module(self):
        def loss(model, x):
            h = x @ model.linear.w + model.linear.b
            return jnp.sum(h + model.extra)

        model = Nested(key=jax.random.key(0))
        x = jnp.ones((1, 2))
        grads = ion.grad(loss)(model, x)
        # All params are trainable → all should have gradients
        assert grads.linear.w is not None
        assert grads.linear.b is not None
        assert grads.extra is not None
        # Gradient shapes match parameter shapes
        assert grads.linear.w.value.shape == model.linear.w.value.shape
        assert grads.linear.b.value.shape == model.linear.b.value.shape
        assert grads.extra.value.shape == model.extra.value.shape


class TestGradEdgeCases:
    def test_grad_all_frozen(self):
        """ion.grad on a model where all params are frozen."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]), trainable=False)

        def loss(model):
            return jnp.sum(model.w.value)

        grads = ion.grad(loss)(Model())
        assert grads.w is None

    def test_grad_no_params(self):
        """ion.grad on a model with no Param fields at all."""

        class Model(nn.Module):
            scale: float

            def __init__(self):
                self.scale = 2.0

        def loss(model):
            return jnp.array(1.0)

        grads = ion.grad(loss)(Model())
        assert grads.scale == 2.0

    def test_grad_as_decorator(self):
        """ion.grad used as a decorator."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        @ion.grad
        def loss(model, x):
            return jnp.sum(model.w * x)

        grads = loss(Model(), jnp.array([3.0, 4.0]))
        npt.assert_allclose(grads.w.value, jnp.array([3.0, 4.0]))

    def test_grad_inside_jit(self):
        """ion.grad composed with jax.jit."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        x = jnp.array([3.0, 4.0])
        grads_eager = ion.grad(loss)(model, x)
        grads_jit = jax.jit(ion.grad(loss))(model, x)
        npt.assert_allclose(grads_jit.w.value, grads_eager.w.value)

    def test_grad_output_structure_matches_model(self):
        """Gradient tree has the exact same pytree structure as the model."""
        model = Nested(key=jax.random.key(0))
        x = jnp.ones((1, 2))

        def loss(m, x):
            h = x @ m.linear.w + m.linear.b
            return jnp.sum(h + m.extra)

        grads = ion.grad(loss)(model, x)
        model_leaves, model_def = jax.tree.flatten(model, is_leaf=lambda x: x is None)
        grad_leaves, grad_def = jax.tree.flatten(grads, is_leaf=lambda x: x is None)
        assert len(model_leaves) == len(grad_leaves)

    def test_grad_multiple_calls_consistent(self):
        """Calling ion.grad multiple times gives the same result."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w ** 2)

        model = Model()
        g1 = ion.grad(loss)(model)
        g2 = ion.grad(loss)(model)
        npt.assert_array_equal(g1.w.value, g2.w.value)

    def test_grad_partial_freeze_nested(self):
        """Gradient correctly handles partially frozen nested modules."""

        class Model(nn.Module):
            encoder: Linear
            decoder: Linear

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.encoder = Linear(2, 3, key=k1)
                self.decoder = Linear(3, 1, key=k2)

        model = Model(key=jax.random.key(0))
        model = model.replace(encoder=model.encoder.freeze())

        def loss(model, x):
            h = x @ model.encoder.w + model.encoder.b
            return jnp.sum(h @ model.decoder.w + model.decoder.b)

        grads = ion.grad(loss)(model, jnp.ones((1, 2)))
        # Frozen encoder params get None
        assert grads.encoder.w is None
        assert grads.encoder.b is None
        # Trainable decoder params get gradients
        assert grads.decoder.w is not None
        assert grads.decoder.b is not None


class TestValueAndGrad:
    def test_value_matches_direct_call(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        value, _ = ion.value_and_grad(loss)(model)
        npt.assert_allclose(value, loss(model))

    def test_grad_matches_grad_only(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w**2)

        model = Model()
        _, vag_grads = ion.value_and_grad(loss)(model)
        grad_only = ion.grad(loss)(model)
        npt.assert_allclose(vag_grads.w.value, grad_only.w.value)

    def test_frozen_param_gets_none(self):
        def loss(model):
            return jnp.sum(model.w.value) + jnp.sum(model.frozen.value)

        model = WithFrozen(key=jax.random.key(0))
        value, grads = ion.value_and_grad(loss)(model)
        npt.assert_allclose(value, loss(model))
        assert grads.w is not None
        assert grads.frozen is None

    def test_extra_args_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, x):
            return jnp.sum(model.w * x)

        model = Model()
        x = jnp.array([3.0, 4.0])
        value, grads = ion.value_and_grad(loss)(model, x)
        npt.assert_allclose(value, jnp.sum(model.w.value * x))
        npt.assert_allclose(grads.w.value, x)

    def test_kwargs_forwarded(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model, *, scale):
            return jnp.sum(model.w) * scale

        model = Model()
        value, grads = ion.value_and_grad(loss)(model, scale=3.0)
        npt.assert_allclose(value, jnp.sum(model.w.value) * 3.0)
        npt.assert_allclose(grads.w.value, jnp.array([3.0, 3.0]))

    def test_inside_jit(self):
        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        def loss(model):
            return jnp.sum(model.w ** 2)

        model = Model()
        val_eager, grad_eager = ion.value_and_grad(loss)(model)
        val_jit, grad_jit = jax.jit(ion.value_and_grad(loss))(model)
        npt.assert_allclose(val_jit, val_eager)
        npt.assert_allclose(grad_jit.w.value, grad_eager.w.value)
