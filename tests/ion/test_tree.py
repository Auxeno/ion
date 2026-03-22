import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn, tree
from ion.nn.module import _Static


class TestFreeze:
    def test_freeze_sets_trainable_false(self):
        """Freezing sets all Param trainable flags to False."""
        data = {"w": nn.Param(jnp.ones(2)), "b": nn.Param(jnp.zeros(2))}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        assert frozen["b"].trainable is False

    def test_freeze_preserves_values(self):
        """Freezing preserves the underlying array values."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        frozen = tree.freeze(data)
        npt.assert_array_equal(frozen["w"]._value, jnp.array([1.0, 2.0]))

    def test_freeze_idempotent(self):
        """Freezing an already-frozen Param is a no-op."""
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        npt.assert_array_equal(frozen["w"]._value, jnp.ones(2))

    def test_freeze_on_module(self):
        """Freezing a Module sets all its Params to non-trainable."""
        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3))

        m = Model()
        frozen = tree.freeze(m)
        assert frozen.w.trainable is False
        assert frozen.b.trainable is False

    def test_freeze_nested_module(self):
        """Freezing recursively freezes Params in nested Modules."""
        class Inner(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))

        class Outer(nn.Module):
            inner: Inner
            b: nn.Param

            def __init__(self):
                self.inner = Inner()
                self.b = nn.Param(jnp.zeros(2))

        m = Outer()
        frozen = tree.freeze(m)
        assert frozen.inner.w.trainable is False
        assert frozen.b.trainable is False

    def test_freeze_does_not_affect_non_params(self):
        """Freezing leaves non-Param fields unchanged."""
        class Model(nn.Module):
            w: nn.Param
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.scale = 2.0

        m = Model()
        frozen = tree.freeze(m)
        assert frozen.scale == 2.0


class TestUnfreeze:
    def test_unfreeze_sets_trainable_true(self):
        """Unfreezing sets trainable flag to True."""
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_unfreeze_preserves_values(self):
        """Unfreezing preserves the underlying array values."""
        data = {"w": nn.Param(jnp.array([3.0, 4.0]), trainable=False)}
        unfrozen = tree.unfreeze(data)
        npt.assert_array_equal(unfrozen["w"]._value, jnp.array([3.0, 4.0]))

    def test_unfreeze_idempotent(self):
        """Unfreezing an already-trainable Param is a no-op."""
        data = {"w": nn.Param(jnp.ones(2), trainable=True)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_freeze_then_unfreeze_roundtrip(self):
        """Freeze then unfreeze restores trainable state and values."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        roundtripped = tree.unfreeze(tree.freeze(data))
        assert roundtripped["w"].trainable is True
        npt.assert_array_equal(roundtripped["w"]._value, jnp.array([1.0, 2.0]))


class TestFreezePreservesPlainArrays:
    def test_freeze_preserves_jax_array_field(self):
        """Freeze doesn't affect plain jax.Array fields (only Params)."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (3,)))
                self.buf = jnp.ones(3)

        m = Model(key=jax.random.key(0))
        frozen = tree.freeze(m)
        npt.assert_array_equal(frozen.buf, m.buf)
        assert frozen.w.trainable is False

    def test_unfreeze_preserves_jax_array_field(self):
        """Unfreeze doesn't affect plain jax.Array fields."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (3,)), trainable=False)
                self.buf = jnp.ones(3)

        m = Model(key=jax.random.key(0))
        unfrozen = tree.unfreeze(m)
        npt.assert_array_equal(unfrozen.buf, m.buf)
        assert unfrozen.w.trainable is True


class Test_Static:
    def test_pytree_no_children(self):
        """A _Static node exposes no pytree children."""
        assert jax.tree.leaves(_Static(42)) == []

    def test_value_preserved_through_flatten(self):
        """Flattening and unflattening a _Static preserves its value."""
        s = _Static({"dropout": 0.1, "mode": "train"})
        children, aux = s.tree_flatten()
        assert children == []
        recovered = _Static.tree_unflatten(aux, children)
        assert recovered.value == s.value

    def test_used_in_tree_map(self):
        """tree_map sees no children inside _Static, so the inner value is untouched."""
        pytree = {"arr": jnp.array(1.0), "cfg": _Static(99)}
        mapped = jax.tree.map(lambda x: x * 2, pytree)
        npt.assert_array_equal(mapped["arr"], jnp.array(2.0))
        # _Static has no children so tree_map passes it through unchanged
        assert isinstance(mapped["cfg"], _Static)
        assert mapped["cfg"].value == 99


class TestJaxJit:
    def test_jax_jit_with_mixed_leaves(self):
        """jax.jit works with modules that have mixed array/non-array fields."""
        from collections.abc import Callable

        class MixedLeaves(nn.Module):
            w: nn.Param
            scale: int
            act: Callable

            def __init__(self, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (2, 2)))
                self.scale = 3
                self.act = jax.nn.relu

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

        keys = jax.random.split(jax.random.key(0), 2)
        model = nn.Sequential(
            nn.Linear(2, 4, key=keys[0]),
            jax.nn.relu,
            nn.Linear(4, 1, key=keys[1]),
        )

        def fn(model, x):
            return model(x)

        x = jnp.ones((1, 2))
        eager = fn(model, x)
        jitted = jax.jit(fn)(model, x)
        npt.assert_allclose(jitted, eager)


class TestAstype:
    def test_astype_param_to_bfloat16(self):
        """Casting a float32 Param to bfloat16 updates its dtype."""
        data = {"w": nn.Param(jnp.ones(3, dtype=jnp.float32))}
        result = tree.astype(data, jnp.bfloat16)
        assert result["w"]._value.dtype == jnp.bfloat16
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is True

    def test_astype_preserves_trainable_flag(self):
        """Casting preserves the Param trainable flag."""
        data = {"w": nn.Param(jnp.ones(3), trainable=False)}
        result = tree.astype(data, jnp.bfloat16)
        assert result["w"].trainable is False
        assert result["w"]._value.dtype == jnp.bfloat16

    def test_astype_plain_float_array(self):
        """Casting converts plain float arrays to the target dtype."""
        data = {"buf": jnp.ones(3, dtype=jnp.float32)}
        result = tree.astype(data, jnp.bfloat16)
        assert result["buf"].dtype == jnp.bfloat16

    def test_astype_leaves_int_arrays_unchanged(self):
        """Casting to a float dtype leaves integer arrays unchanged."""
        data = {"ids": jnp.array([1, 2, 3], dtype=jnp.int32)}
        result = tree.astype(data, jnp.bfloat16)
        assert result["ids"].dtype == jnp.int32

    def test_astype_on_module(self):
        """Casting a Module converts all its float Params and arrays."""
        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(2, dtype=jnp.float32))
                self.buf = jnp.zeros(2, dtype=jnp.float32)

        m = Model()
        result = tree.astype(m, jnp.bfloat16)
        assert result.w._value.dtype == jnp.bfloat16
        assert result.buf.dtype == jnp.bfloat16

    def test_astype_roundtrip(self):
        """Casting f32 to bf16 and back preserves values within tolerance."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0, 3.0]))}
        bf16 = tree.astype(data, jnp.bfloat16)
        f32 = tree.astype(bf16, jnp.float32)
        assert f32["w"]._value.dtype == jnp.float32
        npt.assert_allclose(f32["w"]._value, jnp.array([1.0, 2.0, 3.0]), atol=1e-2)

    def test_astype_nested_module(self):
        """Casting recursively converts Params in nested Modules."""
        class Inner(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))

        class Outer(nn.Module):
            inner: Inner
            b: nn.Param

            def __init__(self):
                self.inner = Inner()
                self.b = nn.Param(jnp.zeros(2))

        m = Outer()
        result = tree.astype(m, jnp.bfloat16)
        assert result.inner.w._value.dtype == jnp.bfloat16
        assert result.b._value.dtype == jnp.bfloat16

    def test_astype_under_jit(self):
        """astype works correctly inside a jit-compiled function."""
        data = {"w": nn.Param(jnp.ones(3))}

        @jax.jit
        def f(d):
            casted = tree.astype(d, jnp.bfloat16)
            return casted["w"]._value

        result = f(data)
        assert result.dtype == jnp.bfloat16

    def test_astype_differentiable(self):
        """Gradients flow through cast (f32 -> bf16 -> f32 roundtrip)."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0, 3.0]))}

        def loss(d):
            bf16 = tree.astype(d, jnp.bfloat16)
            return jnp.sum(bf16["w"]._value.astype(jnp.float32) ** 2)

        grads = jax.grad(loss)(data)
        # d/dw (w^2) = 2w
        npt.assert_allclose(grads["w"]._value, jnp.array([2.0, 4.0, 6.0]), atol=1e-2)
        assert grads["w"]._value.dtype == jnp.float32

    def test_astype_complex_param(self):
        """Casting to a float dtype leaves complex Params unchanged."""
        data = {"w": nn.Param(jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64))}
        result = tree.astype(data, jnp.bfloat16)
        assert result["w"]._value.dtype == jnp.complex64

    def test_astype_complex_array(self):
        """Casting to a float dtype leaves plain complex arrays unchanged."""
        data = {"buf": jnp.array([1 + 0j], dtype=jnp.complex64)}
        result = tree.astype(data, jnp.bfloat16)
        assert result["buf"].dtype == jnp.complex64

    def test_astype_complex_target_leaves_floats_unchanged(self):
        """Casting to a complex dtype touches complex leaves but not float leaves."""
        with jax.enable_x64(True):
            data = {
                "w": nn.Param(jnp.ones(2, dtype=jnp.float32)),
                "z": nn.Param(jnp.array([1 + 2j], dtype=jnp.complex64)),
                "buf": jnp.zeros(2, dtype=jnp.float32),
                "cbuf": jnp.array([3 + 4j], dtype=jnp.complex64),
            }
            result = tree.astype(data, jnp.complex128)
            assert result["w"]._value.dtype == jnp.float32
            assert result["z"]._value.dtype == jnp.complex128
            assert result["buf"].dtype == jnp.float32
            assert result["cbuf"].dtype == jnp.complex128

    def test_astype_int_target_casts_int_leaves(self):
        """Casting to int8 touches integer leaves."""
        data = {
            "ids": nn.Param(jnp.array([1, 2, 3], dtype=jnp.int32)),
            "buf": jnp.array([4, 5], dtype=jnp.int32),
        }
        result = tree.astype(data, jnp.int8)
        assert result["ids"]._value.dtype == jnp.int8
        assert result["buf"].dtype == jnp.int8

    def test_astype_int_target_leaves_floats_unchanged(self):
        """Casting to int8 does not touch float or complex leaves."""
        data = {
            "w": nn.Param(jnp.ones(2, dtype=jnp.float32)),
            "z": nn.Param(jnp.array([1 + 2j], dtype=jnp.complex64)),
            "buf": jnp.zeros(2, dtype=jnp.float32),
        }
        result = tree.astype(data, jnp.int8)
        assert result["w"]._value.dtype == jnp.float32
        assert result["z"]._value.dtype == jnp.complex64
        assert result["buf"].dtype == jnp.float32

    def test_astype_params_only_casts_params(self):
        """params_only=True casts Params but leaves plain arrays unchanged."""
        data = {
            "w": nn.Param(jnp.ones(3, dtype=jnp.float32)),
            "buf": jnp.zeros(3, dtype=jnp.float32),
        }
        result = tree.astype(data, jnp.bfloat16, params_only=True)
        assert result["w"]._value.dtype == jnp.bfloat16
        assert result["buf"].dtype == jnp.float32

    def test_astype_params_only_on_module(self):
        """params_only=True on a module with mixed Param/array fields."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(2, dtype=jnp.float32))
                self.buf = jnp.zeros(2, dtype=jnp.float32)

        m = Model()
        result = tree.astype(m, jnp.bfloat16, params_only=True)
        assert result.w._value.dtype == jnp.bfloat16
        assert result.buf.dtype == jnp.float32


class Test_StaticEdgeCases:
    def test_static_equality_is_identity_based(self):
        """_Static doesn't define __eq__, so equal values aren't equal."""
        s1 = _Static(42)
        s2 = _Static(42)
        assert s1 is not s2
        assert not (s1 == s2)

    def test_static_with_mutable_value(self):
        """_Static holds a reference, so mutating the original mutates the wrapper."""
        data = {"key": "value"}
        s = _Static(data)
        data["key"] = "mutated"
        assert s.value["key"] == "mutated"

    def test_static_in_jit_recompiles_on_value_change(self):
        """Changing a _Static value triggers JIT recompilation."""

        class Model(nn.Module):
            w: nn.Param
            scale: int

            def __init__(self, scale):
                self.w = nn.Param(jnp.ones(2))
                self.scale = scale

        @jax.jit
        def f(model):
            return jnp.sum(model.w) * model.scale

        m1 = Model(scale=2)
        m2 = Model(scale=3)

        r1 = f(m1)
        r2 = f(m2)

        assert float(r1) != float(r2)
        npt.assert_allclose(r1, 4.0)
        npt.assert_allclose(r2, 6.0)
