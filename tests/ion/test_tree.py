import jax.numpy as jnp
import numpy.testing as npt

from ion import nn, tree


class TestApplyUpdates:
    def test_param_update_preserves_wrapper(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        updates = {"w": nn.Param(jnp.array([0.1, 0.2]))}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        assert result["w"].trainable is True
        npt.assert_allclose(result["w"].value, jnp.array([1.1, 2.2]))

    def test_frozen_param_unchanged(self):
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([9.0]))}
        result = tree.apply_updates(data, updates)
        assert result["w"].trainable is False
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_none_update_skipped(self):
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": None}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_plain_array_update(self):
        data = {"x": jnp.array([1.0, 2.0])}
        updates = {"x": jnp.array([0.5, 0.5])}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["x"], jnp.array([1.5, 2.5]))

    def test_raw_array_delta_on_param(self):
        """Update is a plain array (not Param), applied to Param.value."""
        data = {"w": nn.Param(jnp.array([1.0]))}
        updates = {"w": jnp.array([0.5])}
        result = tree.apply_updates(data, updates)
        assert isinstance(result["w"], nn.Param)
        npt.assert_allclose(result["w"].value, jnp.array([1.5]))

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
        npt.assert_allclose(result.w.value, jnp.array([2.0, 2.0]))
        npt.assert_allclose(result.b.value, jnp.zeros(2))


class TestFreeze:
    def test_freeze_sets_trainable_false(self):
        data = {"w": nn.Param(jnp.ones(2)), "b": nn.Param(jnp.zeros(2))}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        assert frozen["b"].trainable is False

    def test_freeze_preserves_values(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        frozen = tree.freeze(data)
        npt.assert_array_equal(frozen["w"].value, jnp.array([1.0, 2.0]))

    def test_freeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=False)}
        frozen = tree.freeze(data)
        assert frozen["w"].trainable is False
        npt.assert_array_equal(frozen["w"].value, jnp.ones(2))

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
        npt.assert_array_equal(unfrozen["w"].value, jnp.array([3.0, 4.0]))

    def test_unfreeze_idempotent(self):
        data = {"w": nn.Param(jnp.ones(2), trainable=True)}
        unfrozen = tree.unfreeze(data)
        assert unfrozen["w"].trainable is True

    def test_freeze_then_unfreeze_roundtrip(self):
        data = {"w": nn.Param(jnp.array([1.0, 2.0]))}
        roundtripped = tree.unfreeze(tree.freeze(data))
        assert roundtripped["w"].trainable is True
        npt.assert_array_equal(roundtripped["w"].value, jnp.array([1.0, 2.0]))


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
        npt.assert_allclose(result["w"].value, jnp.array([1.1, 2.2]))
        npt.assert_allclose(result["b"].value, jnp.array([0.0]))  # unchanged


class TestApplyUpdatesEdgeCases:
    def test_apply_updates_frozen_param_with_large_update_silently_ignored(self):
        """A large update to a frozen param is silently dropped with no warning."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=False)}
        updates = {"w": nn.Param(jnp.array([999999.0]))}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"].value, jnp.array([1.0]))

    def test_apply_updates_update_trainable_flag_ignored(self):
        """The trainable flag on the update Param is ignored; the model's flag wins."""
        data = {"w": nn.Param(jnp.array([1.0]), trainable=True)}
        updates = {"w": nn.Param(jnp.array([0.5]), trainable=False)}
        result = tree.apply_updates(data, updates)
        npt.assert_allclose(result["w"].value, jnp.array([1.5]))
        assert result["w"].trainable is True
