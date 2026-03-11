import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn, tree
from ion.tree import Static


class TestApplyUpdates:
    def test_param_update_preserves_wrapper(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        updates = {"w": nn.Param(jnp.array([0.1, 0.2]))}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is True
        npt.assert_allclose(result["w"]._value, jnp.array([1.1, 2.2]))

    def test_frozen_param_unchanged(self):
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([9.0]))}
        result = tree.apply_updates(data, updates)
        assert result["w"].trainable is False
        npt.assert_allclose(result["w"]._value, jnp.array([1.0]))

    def test_none_update_skipped(self):
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": None}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"]._value, jnp.array([1.0]))

    def test_plain_array_unchanged(self):
        """Non-Param arrays are not modified by apply_updates."""
        data = {"x": jnp.array([1.0, 2.0])}
        updates = {"x": jnp.array([0.5, 0.5])}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["x"], jnp.array([1.0, 2.0]))

    def test_raw_array_delta_on_param(self):
        """Update is a plain array (not Param), applied to Param._value."""
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": jnp.array([0.5])}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"]._value, jnp.array([1.5]))

    def test_on_module(self):
        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.b = nn.Param(jnp.zeros(2))

        model = Model()
        updates = Model()  # all ones/zeros as deltas
        result = tree.apply_updates(model, updates)
        npt.assert_allclose(result.w._value, jnp.array([2.0, 2.0]))
        npt.assert_allclose(result.b._value, jnp.zeros(2))


class TestFreeze:
    def test_freeze_sets_trainable_false(self):
        data = {"w": nn.Param(jnp.ones(2)), "b": nn.Param(jnp.zeros(2))}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        assert frozen["b"].trainable is False

    def test_freeze_preserves_values(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        frozen = tree.freeze(data)
        npt.assert_array_equal(frozen["w"]._value, jnp.array([1.0, 2.0]))

    def test_freeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        npt.assert_array_equal(frozen["w"]._value, jnp.ones(2))

    def test_freeze_on_module(self):
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
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_unfreeze_preserves_values(self):
        data = {"w": nn.Param(jnp.array([3.0, 4.0]), trainable=False)}
        unfrozen = tree.unfreeze(data)
        npt.assert_array_equal(unfrozen["w"]._value, jnp.array([3.0, 4.0]))

    def test_unfreeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=True)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_freeze_then_unfreeze_roundtrip(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        roundtripped = tree.unfreeze(tree.freeze(data))
        assert roundtripped["w"].trainable is True
        npt.assert_array_equal(roundtripped["w"]._value, jnp.array([1.0, 2.0]))


class TestApplyUpdatesExtended:
    def test_apply_updates_preserves_trainable_flag(self):
        """After apply_updates, Param stays trainable."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=True)}
        updates = {"w": jnp.array([0.5])}
        result = tree.apply_updates(data, updates)
        assert result["w"].trainable is True

    def test_apply_updates_mixed_trainable_frozen(self):
        data = {
            "w": nn.Param(jnp.array([1.0, 2.0])),
            "b": nn.Param(jnp.array([0.0]), trainable=False),
        }
        updates = {
            "w": jnp.array([0.1, 0.2]),
            "b": jnp.array([9.0]),
        }
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"]._value, jnp.array([1.1, 2.2]))
        npt.assert_allclose(result["b"]._value, jnp.array([0.0]))  # unchanged


class TestApplyUpdatesEdgeCases:
    def test_apply_updates_frozen_param_with_large_update_silently_ignored(self):
        """A large update to a frozen param is silently dropped with no warning."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([999999.0]))}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"]._value, jnp.array([1.0]))

    def test_apply_updates_update_trainable_flag_ignored(self):
        """The trainable flag on the update Param is ignored; the model's flag wins."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=True)}
        updates = {"w": nn.Param(jnp.array([0.5]), trainable=False)}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"]._value, jnp.array([1.5]))
        assert result["w"].trainable is True

    def test_apply_updates_batchnorm_running_stats_unchanged(self):
        """BatchNorm running stats (non-Param arrays) are not modified."""
        bn = nn.BatchNorm(8, training=True)
        updates = jax.tree.map(jnp.zeros_like, bn)
        result = tree.apply_updates(bn, updates)
        npt.assert_array_equal(result.running_mean, bn.running_mean)
        npt.assert_array_equal(result.running_var, bn.running_var)


class TestApplyUpdatesUnderJit:
    def test_apply_updates_under_jit(self):
        """apply_updates works correctly inside jax.jit."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3))

        model = Model()
        updates = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, model)

        result = jax.jit(tree.apply_updates)(model, updates)
        npt.assert_allclose(result.w._value, jnp.array([1.1, 1.1, 1.1]))
        npt.assert_allclose(result.b._value, jnp.array([0.1, 0.1, 0.1]))

    def test_apply_updates_skips_frozen_under_jit(self):
        """Frozen params are still skipped by apply_updates inside jax.jit."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3), trainable=False)

        model = Model()
        updates = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, model)

        result = jax.jit(tree.apply_updates)(model, updates)
        npt.assert_allclose(result.w._value, jnp.array([1.1, 1.1, 1.1]))
        npt.assert_allclose(result.b._value, jnp.zeros(3))  # unchanged


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


class TestCast:
    def test_cast_param_to_bfloat16(self):
        data = {"w": nn.Param(jnp.ones(3, dtype=jnp.float32))}
        result = tree.cast(data, jnp.bfloat16)
        assert result["w"]._value.dtype == jnp.bfloat16
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is True

    def test_cast_preserves_trainable_flag(self):
        data = {"w": nn.Param(jnp.ones(3), trainable=False)}
        result = tree.cast(data, jnp.bfloat16)
        assert result["w"].trainable is False
        assert result["w"]._value.dtype == jnp.bfloat16

    def test_cast_plain_float_array(self):
        data = {"buf": jnp.ones(3, dtype=jnp.float32)}
        result = tree.cast(data, jnp.bfloat16)
        assert result["buf"].dtype == jnp.bfloat16

    def test_cast_leaves_int_arrays_unchanged(self):
        data = {"ids": jnp.array([1, 2, 3], dtype=jnp.int32)}
        result = tree.cast(data, jnp.bfloat16)
        assert result["ids"].dtype == jnp.int32

    def test_cast_on_module(self):
        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(2, dtype=jnp.float32))
                self.buf = jnp.zeros(2, dtype=jnp.float32)

        m = Model()
        result = tree.cast(m, jnp.bfloat16)
        assert result.w._value.dtype == jnp.bfloat16
        assert result.buf.dtype == jnp.bfloat16

    def test_cast_roundtrip(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0, 3.0]))}
        bf16 = tree.cast(data, jnp.bfloat16)
        f32 = tree.cast(bf16, jnp.float32)
        assert f32["w"]._value.dtype == jnp.float32
        npt.assert_allclose(f32["w"]._value, jnp.array([1.0, 2.0, 3.0]), atol=1e-2)

    def test_cast_nested_module(self):
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
        result = tree.cast(m, jnp.bfloat16)
        assert result.inner.w._value.dtype == jnp.bfloat16
        assert result.b._value.dtype == jnp.bfloat16

    def test_cast_under_jit(self):
        data = {"w": nn.Param(jnp.ones(3))}

        @jax.jit
        def f(d):
            casted = tree.cast(d, jnp.bfloat16)
            return casted["w"]._value

        result = f(data)
        assert result.dtype == jnp.bfloat16

    def test_cast_differentiable(self):
        """Gradients flow through cast (f32 -> bf16 -> f32 roundtrip)."""
        data = {"w": nn.Param(jnp.array([1.0, 2.0, 3.0]))}

        def loss(d):
            bf16 = tree.cast(d, jnp.bfloat16)
            return jnp.sum(bf16["w"]._value.astype(jnp.float32) ** 2)

        grads = jax.grad(loss)(data)
        # d/dw (w^2) = 2w
        npt.assert_allclose(grads["w"]._value, jnp.array([2.0, 4.0, 6.0]), atol=1e-2)
        assert grads["w"]._value.dtype == jnp.float32

    def test_cast_complex_param(self):
        """Casting to a float dtype leaves complex Params unchanged."""
        data = {"w": nn.Param(jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64))}
        result = tree.cast(data, jnp.bfloat16)
        assert result["w"]._value.dtype == jnp.complex64

    def test_cast_complex_array(self):
        """Casting to a float dtype leaves plain complex arrays unchanged."""
        data = {"buf": jnp.array([1 + 0j], dtype=jnp.complex64)}
        result = tree.cast(data, jnp.bfloat16)
        assert result["buf"].dtype == jnp.complex64

    def test_cast_complex_target_leaves_floats_unchanged(self):
        """Casting to a complex dtype touches complex leaves but not float leaves."""
        with jax.enable_x64(True):
            data = {
                "w": nn.Param(jnp.ones(2, dtype=jnp.float32)),
                "z": nn.Param(jnp.array([1 + 2j], dtype=jnp.complex64)),
                "buf": jnp.zeros(2, dtype=jnp.float32),
                "cbuf": jnp.array([3 + 4j], dtype=jnp.complex64),
            }
            result = tree.cast(data, jnp.complex128)
            assert result["w"]._value.dtype == jnp.float32
            assert result["z"]._value.dtype == jnp.complex128
            assert result["buf"].dtype == jnp.float32
            assert result["cbuf"].dtype == jnp.complex128

    def test_cast_int_target_casts_int_leaves(self):
        """Casting to int8 touches integer leaves."""
        data = {
            "ids": nn.Param(jnp.array([1, 2, 3], dtype=jnp.int32)),
            "buf": jnp.array([4, 5], dtype=jnp.int32),
        }
        result = tree.cast(data, jnp.int8)
        assert result["ids"]._value.dtype == jnp.int8
        assert result["buf"].dtype == jnp.int8

    def test_cast_int_target_leaves_floats_unchanged(self):
        """Casting to int8 does not touch float or complex leaves."""
        data = {
            "w": nn.Param(jnp.ones(2, dtype=jnp.float32)),
            "z": nn.Param(jnp.array([1 + 2j], dtype=jnp.complex64)),
            "buf": jnp.zeros(2, dtype=jnp.float32),
        }
        result = tree.cast(data, jnp.int8)
        assert result["w"]._value.dtype == jnp.float32
        assert result["z"]._value.dtype == jnp.complex64
        assert result["buf"].dtype == jnp.float32

    def test_cast_params_only_casts_params(self):
        """params_only=True casts Params but leaves plain arrays unchanged."""
        data = {
            "w": nn.Param(jnp.ones(3, dtype=jnp.float32)),
            "buf": jnp.zeros(3, dtype=jnp.float32),
        }
        result = tree.cast(data, jnp.bfloat16, params_only=True)
        assert result["w"]._value.dtype == jnp.bfloat16
        assert result["buf"].dtype == jnp.float32

    def test_cast_params_only_on_module(self):
        """params_only=True on a module with mixed Param/array fields."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(2, dtype=jnp.float32))
                self.buf = jnp.zeros(2, dtype=jnp.float32)

        m = Model()
        result = tree.cast(m, jnp.bfloat16, params_only=True)
        assert result.w._value.dtype == jnp.bfloat16
        assert result.buf.dtype == jnp.float32



class TestStaticEdgeCases:
    def test_static_equality_is_identity_based(self):
        """Static doesn't define __eq__, so equal values aren't equal."""
        s1 = Static(42)
        s2 = Static(42)
        assert s1 is not s2
        assert not (s1 == s2)

    def test_static_with_mutable_value(self):
        """Static holds a reference, so mutating the original mutates the wrapper."""
        data = {"key": "value"}
        s = Static(data)
        data["key"] = "mutated"
        assert s.value["key"] == "mutated"

    def test_static_in_jit_recompiles_on_value_change(self):
        """Changing a Static value triggers JIT recompilation."""

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
