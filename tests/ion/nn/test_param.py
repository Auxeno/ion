import dataclasses
import tempfile

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

import ion
from ion import nn, tree


class TestPytreeRegistration:
    def test_flatten_unflatten_roundtrip(self):
        """Param survives pytree flatten/unflatten."""
        p = nn.Param(jnp.array([1.0, 2.0, 3.0]))
        leaves, treedef = jtu.tree_flatten(p)
        assert len(leaves) == 1
        npt.assert_array_equal(leaves[0], jnp.array([1.0, 2.0, 3.0]))
        reconstructed = treedef.unflatten(leaves)
        assert isinstance(reconstructed, nn.Param)
        assert reconstructed.trainable is True
        npt.assert_array_equal(reconstructed.value, p.value)

    def test_trainable_preserved_as_static(self):
        """trainable flag is preserved through flatten/unflatten as aux data."""
        p = nn.Param(jnp.array(1.0), trainable=False)
        leaves, treedef = jtu.tree_flatten(p)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.trainable is False

    def test_tree_map_transforms_value(self):
        """jax.tree.map applies to the inner value."""
        p = nn.Param(jnp.array([1.0, 2.0]))
        doubled = jax.tree.map(lambda x: x * 2, p)
        assert isinstance(doubled, nn.Param)
        npt.assert_array_equal(doubled.value, jnp.array([2.0, 4.0]))


class TestJaxArrayProtocol:
    def test_jax_array_protocol(self):
        """__jax_array__ lets jnp operations work directly on Param."""
        p = nn.Param(jnp.array([1.0, 2.0, 3.0]))
        result = jnp.sum(p)  # type: ignore[arg-type]
        npt.assert_allclose(result, 6.0)

    def test_jnp_dot(self):
        """Param works with jnp.dot."""
        a = nn.Param(jnp.array([1.0, 2.0]))
        b = jnp.array([3.0, 4.0])
        npt.assert_allclose(jnp.dot(a, b), 11.0)  # type: ignore[arg-type]


class TestArithmetic:
    def test_add(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = p + jnp.array([3.0, 4.0])
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([4.0, 6.0]))

    def test_radd(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = jnp.array([3.0, 4.0]) + p
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([4.0, 6.0]))

    def test_sub(self):
        p = nn.Param(jnp.array([5.0, 6.0]))
        result = p - jnp.array([1.0, 2.0])
        npt.assert_array_equal(result, jnp.array([4.0, 4.0]))

    def test_mul(self):
        p = nn.Param(jnp.array([2.0, 3.0]))
        result = p * jnp.array([4.0, 5.0])
        npt.assert_array_equal(result, jnp.array([8.0, 15.0]))

    def test_matmul(self):
        p = nn.Param(jnp.eye(2))
        x = jnp.array([1.0, 2.0])
        npt.assert_array_equal(p @ x, x)

    def test_neg(self):
        p = nn.Param(jnp.array([1.0, -2.0]))
        result = -p
        npt.assert_array_equal(result, jnp.array([-1.0, 2.0]))

    def test_param_to_param_add(self):
        """Arithmetic between two Params unwraps both."""
        a = nn.Param(jnp.array([1.0]))
        b = nn.Param(jnp.array([2.0]))
        result = a + b
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([3.0]))

    def test_rsub(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = jnp.array([10.0, 10.0]) - p
        npt.assert_array_equal(result, jnp.array([9.0, 8.0]))

    def test_rmul(self):
        p = nn.Param(jnp.array([2.0, 3.0]))
        result = jnp.array([4.0, 5.0]) * p
        npt.assert_array_equal(result, jnp.array([8.0, 15.0]))

    def test_truediv(self):
        p = nn.Param(jnp.array([6.0, 8.0]))
        result = p / jnp.array([2.0, 4.0])
        npt.assert_array_equal(result, jnp.array([3.0, 2.0]))

    def test_rtruediv(self):
        p = nn.Param(jnp.array([2.0, 4.0]))
        result = jnp.array([6.0, 8.0]) / p
        npt.assert_array_equal(result, jnp.array([3.0, 2.0]))

    def test_floordiv(self):
        p = nn.Param(jnp.array([7.0, 9.0]))
        result = p // jnp.array([2.0, 4.0])
        npt.assert_array_equal(result, jnp.array([3.0, 2.0]))

    def test_rfloordiv(self):
        p = nn.Param(jnp.array([2.0, 4.0]))
        result = jnp.array([7.0, 9.0]) // p
        npt.assert_array_equal(result, jnp.array([3.0, 2.0]))

    def test_mod(self):
        p = nn.Param(jnp.array([7.0, 9.0]))
        result = p % jnp.array([3.0, 4.0])
        npt.assert_array_equal(result, jnp.array([1.0, 1.0]))

    def test_rmod(self):
        p = nn.Param(jnp.array([3.0, 4.0]))
        result = jnp.array([7.0, 9.0]) % p
        npt.assert_array_equal(result, jnp.array([1.0, 1.0]))

    def test_pow(self):
        p = nn.Param(jnp.array([2.0, 3.0]))
        result = p ** jnp.array([3.0, 2.0])
        npt.assert_array_equal(result, jnp.array([8.0, 9.0]))

    def test_rpow(self):
        p = nn.Param(jnp.array([3.0, 2.0]))
        result = jnp.array([2.0, 3.0]) ** p
        npt.assert_array_equal(result, jnp.array([8.0, 9.0]))

    def test_rmatmul(self):
        p = nn.Param(jnp.eye(2))
        x = jnp.array([[1.0, 2.0]])
        npt.assert_array_equal(x @ p, x)

    def test_pos(self):
        p = nn.Param(jnp.array([-1.0, 2.0]))
        result = +p
        npt.assert_array_equal(result, jnp.array([-1.0, 2.0]))

    def test_abs(self):
        p = nn.Param(jnp.array([-3.0, 2.0]))
        result = abs(p)
        npt.assert_array_equal(result, jnp.array([3.0, 2.0]))

    def test_param_to_param_sub(self):
        a = nn.Param(jnp.array([5.0]))
        b = nn.Param(jnp.array([2.0]))
        result = a - b
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([3.0]))

    def test_param_to_param_mul(self):
        a = nn.Param(jnp.array([3.0]))
        b = nn.Param(jnp.array([4.0]))
        result = a * b
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([12.0]))

    def test_param_to_param_matmul(self):
        a = nn.Param(jnp.eye(2))
        b = nn.Param(jnp.array([[1.0], [2.0]]))
        result = a @ b
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([[1.0], [2.0]]))


class TestAttributeForwarding:
    def test_shape(self):
        p = nn.Param(jnp.zeros((3, 4)))
        assert p.shape == (3, 4)

    def test_dtype(self):
        p = nn.Param(jnp.zeros(5, dtype=jnp.float32))
        assert p.dtype == jnp.float32

    def test_getitem(self):
        p = nn.Param(jnp.array([10.0, 20.0, 30.0]))
        npt.assert_allclose(p[1], 20.0)


class TestImmutability:
    def test_frozen_setattr_raises(self):
        """Setting an attribute on a frozen Param raises."""
        p = nn.Param(jnp.array(1.0))
        try:
            p.value = jnp.array(2.0)  # type: ignore[misc]
            assert False, "Should have raised"
        except (AttributeError, dataclasses.FrozenInstanceError):
            pass


class TestJaxTransforms:
    def test_jit(self):
        """Param works through jax.jit."""
        p = nn.Param(jnp.array([1.0, 2.0]))

        @jax.jit
        def f(param):
            return jnp.sum(param.value)

        npt.assert_allclose(f(p), 3.0)

    def test_grad(self):
        """jax.grad flows through Param's value."""
        p = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(param):
            return jnp.sum(param.value**2)

        grads = jax.grad(loss)(p)
        assert isinstance(grads, nn.Param)
        npt.assert_array_equal(grads.value, jnp.array([2.0, 4.0, 6.0]))

    def test_vmap(self):
        """Param works through jax.vmap."""
        vals = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        p = nn.Param(vals)

        @jax.vmap
        def f(param):
            return jnp.sum(param.value)

        result = f(p)
        npt.assert_array_equal(result, jnp.array([3.0, 7.0]))


class TestLuxTransforms:
    def test_ion_grad(self):
        """ion.grad differentiates through Param-containing modules."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))

            def __call__(self, x):
                return jnp.sum(self.w.value * x)

        model = Model(key=jax.random.key(0))
        x = jnp.array([1.0, 2.0])

        grads = ion.grad(lambda m, x: m(x))(model, x)
        # Gradient of sum(w * x) w.r.t. w is x
        npt.assert_allclose(grads.w.value, x)

    def test_ion_value_and_grad(self):
        """ion.value_and_grad works with Param-containing modules."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))

            def __call__(self, x):
                return jnp.sum(self.w.value * x)

        model = Model(key=jax.random.key(0))
        x = jnp.array([1.0, 2.0])

        val, grads = ion.value_and_grad(lambda m, x: m(x))(model, x)
        expected_val = jnp.sum(model.w.value * x)
        npt.assert_allclose(val, expected_val)
        npt.assert_allclose(grads.w.value, x)


class TestTreeUtilities:
    def test_module_params_filters_for_params(self):
        """Module.params keeps only Param leaves."""

        class Model(nn.Module):
            w: nn.Param
            scale: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (3,)))
                self.scale = jnp.array(1.0)

        model = Model(key=jax.random.key(0))
        params = model.params

        assert isinstance(params.w, nn.Param)
        assert params.scale is None

    def test_is_param_predicate(self):
        assert tree.is_param(nn.Param(jnp.array(1.0))) is True
        assert tree.is_param(nn.Param(jnp.array(1.0), trainable=False)) is True
        assert tree.is_param(jnp.array(1.0)) is False
        assert tree.is_param(42) is False

    def test_is_trainable_param_predicate(self):
        assert tree.is_trainable_param(nn.Param(jnp.array(1.0))) is True
        assert tree.is_trainable_param(nn.Param(jnp.array(1.0), trainable=False)) is False
        assert tree.is_trainable_param(jnp.array(1.0)) is False


class TestApplyUpdates:
    def test_apply_updates_preserves_param(self):
        """apply_updates preserves Param structure."""
        p = nn.Param(jnp.array([1.0, 2.0]))
        data = {"w": p}
        updates = {"w": nn.Param(jnp.array([0.1, 0.2]))}

        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.1, 2.2]))

    def test_apply_updates_skips_frozen_param(self):
        """apply_updates leaves frozen Params unchanged."""
        p = nn.Param(jnp.array([1.0, 2.0]), trainable=False)
        data = {"w": p}
        updates = {"w": nn.Param(jnp.array([0.1, 0.2]))}

        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is False
        npt.assert_allclose(result["w"].value, jnp.array([1.0, 2.0]))

    def test_apply_updates_skips_none_updates(self):
        """apply_updates leaves params unchanged when update is None."""
        data = {"w": nn.Param(jnp.array([1.0])), "x": jnp.array(5.0)}
        updates = {"w": None, "x": None}

        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))
        npt.assert_allclose(result["x"], 5.0)


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        """Param-containing models survive save/load."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        model = Model(key=jax.random.key(42))

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            tree.save(f.name, model)
            loaded = tree.load(f.name, model)

        assert isinstance(loaded.w, nn.Param)
        npt.assert_array_equal(loaded.w.value, model.w.value)
        assert loaded.w.trainable is True


class TestComparisons:
    def test_eq(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = p == jnp.array([1.0, 3.0])
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_ne(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = p != jnp.array([1.0, 3.0])
        npt.assert_array_equal(result, jnp.array([False, True]))

    def test_lt(self):
        p = nn.Param(jnp.array([1.0, 3.0]))
        result = p < jnp.array([2.0, 2.0])
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_le(self):
        p = nn.Param(jnp.array([1.0, 2.0]))
        result = p <= jnp.array([1.0, 1.0])
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_gt(self):
        p = nn.Param(jnp.array([3.0, 1.0]))
        result = p > jnp.array([2.0, 2.0])
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_ge(self):
        p = nn.Param(jnp.array([2.0, 1.0]))
        result = p >= jnp.array([2.0, 2.0])
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_eq_param_to_param(self):
        a = nn.Param(jnp.array([1.0, 2.0]))
        b = nn.Param(jnp.array([1.0, 3.0]))
        result = a == b
        assert not isinstance(result, nn.Param)
        npt.assert_array_equal(result, jnp.array([True, False]))

    def test_comparison_returns_array_not_param(self):
        p = nn.Param(jnp.array([1.0]))
        result = p < jnp.array([2.0])
        assert not isinstance(result, nn.Param)
        assert isinstance(result, jax.Array)


class TestProtocols:
    def test_bool_scalar(self):
        p = nn.Param(jnp.array(1.0))
        assert bool(p) is True

    def test_bool_zero(self):
        p = nn.Param(jnp.array(0.0))
        assert bool(p) is False

    def test_len(self):
        p = nn.Param(jnp.array([1.0, 2.0, 3.0]))
        assert len(p) == 3

    def test_iter(self):
        p = nn.Param(jnp.array([10.0, 20.0, 30.0]))
        values = list(p)
        assert len(values) == 3
        npt.assert_allclose(float(values[0]), 10.0)
        npt.assert_allclose(float(values[1]), 20.0)
        npt.assert_allclose(float(values[2]), 30.0)

    def test_getitem_slice(self):
        p = nn.Param(jnp.array([10.0, 20.0, 30.0, 40.0]))
        result = p[1:3]
        npt.assert_array_equal(result, jnp.array([20.0, 30.0]))

    def test_getitem_2d(self):
        p = nn.Param(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_array_equal(p[0], jnp.array([1.0, 2.0]))
        npt.assert_allclose(float(p[1, 1]), 4.0)


class TestAttributeForwardingExtended:
    def test_ndim(self):
        p = nn.Param(jnp.zeros((2, 3, 4)))
        assert p.ndim == 3

    def test_size(self):
        p = nn.Param(jnp.zeros((2, 3)))
        assert p.size == 6

    def test_T(self):
        p = nn.Param(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_array_equal(p.T, jnp.array([[1.0, 3.0], [2.0, 4.0]]))


class TestRepr:
    def test_param_repr(self):
        p = nn.Param(jnp.zeros((3, 4), dtype=jnp.float32))
        assert repr(p) == "Param(f32[3, 4], trainable=True)"

    def test_frozen_param_repr(self):
        p = nn.Param(jnp.zeros(5), trainable=False)
        assert "trainable=False" in repr(p)

    def test_abbreviated_dtypes(self):
        cases = [
            (jnp.float16, "f16"),
            (jnp.float32, "f32"),
            (jnp.bfloat16, "bf16"),
            (jnp.int8, "i8"),
            (jnp.int32, "i32"),
            (jnp.uint8, "u8"),
        ]
        for dtype, abbrev in cases:
            p = nn.Param(jnp.zeros(2, dtype=dtype))
            assert abbrev in repr(p), f"Expected {abbrev} in repr for {dtype}"

    def test_scalar_param_repr(self):
        p = nn.Param(jnp.array(1.0))
        assert "f32[]" in repr(p)

    def test_module_repr_with_param(self):
        """Module __repr__ displays Param wrapper."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.zeros((2, 3)))

        m = Model()
        r = repr(m)
        assert "Param(" in r
