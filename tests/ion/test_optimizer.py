import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

import ion
from ion import nn


class WithBareArray(nn.Module):
    w: nn.Param
    buf: jax.Array

    def __init__(self, key: jax.Array):
        self.w = nn.Param(jax.random.normal(key, (4,)))
        self.buf = jnp.ones(4)

    def __call__(self, x):
        return self.w * x + self.buf


class TestOptimizerInit:
    def test_creates_state(self):
        """Creating an optimizer initializes non-empty internal state."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert optimizer.state is not None
        assert len(jax.tree.leaves(optimizer.state)) > 0

    def test_step_is_zero_int32(self):
        """Initial step counter is zero with int32 dtype."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert optimizer.step == 0
        assert optimizer.step.dtype == jnp.int32

    def test_auto_partitions_frozen(self):
        """Optimizer auto-partitions when all params are frozen."""
        model = nn.Linear(4, 2, key=jax.random.key(0)).freeze()
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert optimizer.state is not None

    def test_no_partition_all_trainable(self):
        """Optimizer skips partitioning when all params are trainable."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert optimizer.state is not None

    def test_model_with_bare_array_fields(self):
        """Non-Param jax.Array fields don't crash init."""
        model = WithBareArray(key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        assert optimizer.state is not None

    def test_bare_array_with_frozen_param(self):
        """Partition handles models with both frozen Params and non-Param arrays."""
        model = WithBareArray(key=jax.random.key(0))
        frozen = model.replace(w=nn.Param(model.w._value, trainable=False))
        optimizer = ion.Optimizer(optax.adam(1e-3), frozen)
        assert optimizer.state is not None


class TestOptimizerUpdate:
    def test_returns_model_and_optimizer(self):
        """Update returns a new model and optimizer instance."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)

        grads = jax.grad(lambda m: jnp.mean(m(jnp.ones((2, 4))) ** 2))(model)
        new_model, new_optimizer = optimizer.update(model, grads)
        assert isinstance(new_model, type(model))
        assert isinstance(new_optimizer, ion.Optimizer)

    def test_increments_step(self):
        """Step counter increments by one after each update."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        x = jnp.ones((2, 4))

        grads = jax.grad(lambda m: jnp.mean(m(x) ** 2))(model)
        model, optimizer = optimizer.update(model, grads)
        assert optimizer.step == 1
        grads = jax.grad(lambda m: jnp.mean(m(x) ** 2))(model)
        model, optimizer = optimizer.update(model, grads)
        assert optimizer.step == 2

    def test_decreases_loss(self):
        """10-step training loop decreases MSE."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        initial_loss = loss_fn(model, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(model, x, y)
            model, optimizer = optimizer.update(model, grads)

        assert loss_fn(model, x, y) < initial_loss

    def test_preserves_frozen_params(self):
        """Frozen params are unchanged after update."""
        model = nn.Linear(4, 2, key=jax.random.key(0)).freeze()
        frozen_w = model.w._value.copy()
        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        grads = jax.grad(lambda m: jnp.mean(m(jnp.ones((2, 4))) ** 2))(model)
        new_model, _ = optimizer.update(model, grads)
        npt.assert_array_equal(new_model.w._value, frozen_w)

    def test_skips_non_param_array_leaves(self):
        """Plain jax.Array fields pass through unchanged."""
        model = WithBareArray(key=jax.random.key(0))
        buf_before = model.buf.copy()
        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        grads = jax.grad(lambda m: jnp.sum(m(jnp.ones(4))))(model)
        new_model, _ = optimizer.update(model, grads)
        npt.assert_array_equal(new_model.buf, buf_before)

    def test_bare_array_with_frozen_param_update(self):
        """Update works with frozen Params and non-Param arrays together."""
        model = WithBareArray(key=jax.random.key(0))
        frozen = model.replace(w=nn.Param(model.w._value, trainable=False))
        frozen_w = frozen.w._value.copy()
        buf_before = frozen.buf.copy()
        optimizer = ion.Optimizer(optax.adam(1e-2), frozen)

        grads = jax.grad(lambda m: jnp.sum(m(jnp.ones(4))))(frozen)
        new_model, _ = optimizer.update(frozen, grads)
        npt.assert_array_equal(new_model.w._value, frozen_w)
        npt.assert_array_equal(new_model.buf, buf_before)

    def test_chained_transforms(self):
        """optax.chain(clip, adam) works with Optimizer."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        transform = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
        optimizer = ion.Optimizer(transform, model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        initial_loss = loss_fn(model, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(model, x, y)
            model, optimizer = optimizer.update(model, grads)

        assert loss_fn(model, x, y) < initial_loss

    def test_kwargs_forwarded(self):
        """Extra kwargs reach the optax transform (e.g. adamw params)."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adamw(1e-3), model)

        grads = jax.grad(lambda m: jnp.mean(m(jnp.ones((2, 4))) ** 2))(model)
        # Should not raise -- adamw accepts params as 3rd arg
        _, new_optimizer = optimizer.update(model, grads)
        assert new_optimizer.step == 1

    def test_kwargs_affect_update_result(self):
        """Extra kwargs change optimizer behavior, not just avoid crashing."""

        def _scaled_sgd(lr):
            """SGD whose updates are multiplied by a `scale` kwarg."""
            base = optax.sgd(lr)

            def init_fn(params):
                return base.init(params)

            def update_fn(updates, state, params=None, *, scale=1.0):
                updates, state = base.update(updates, state, params)
                return jax.tree.map(lambda u: u * scale, updates), state

            return optax.GradientTransformationExtraArgs(init_fn, update_fn)

        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(_scaled_sgd(0.1), model)

        grads = jax.grad(lambda m: jnp.mean(m(jnp.ones((2, 4))) ** 2))(model)

        # scale=0 should leave weights unchanged
        model_zero, _ = optimizer.update(model, grads, scale=0.0)
        npt.assert_array_equal(model_zero.w._value, model.w._value)

        # scale=1 should change weights
        model_one, _ = optimizer.update(model, grads, scale=1.0)
        assert not jnp.array_equal(model_one.w._value, model.w._value)


class TestOptimizerPytree:
    def test_flatten_unflatten_roundtrip(self):
        """Flattening and unflattening preserves step and state."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)

        children, aux = optimizer.tree_flatten()
        restored = ion.Optimizer.tree_unflatten(aux, children)
        assert restored.step == optimizer.step
        for a, b in zip(jax.tree.leaves(restored.state), jax.tree.leaves(optimizer.state)):
            npt.assert_array_equal(a, b)

    def test_jit_compatible(self):
        """Optimizer works inside jax.jit."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        @jax.jit
        def step(model, optimizer, x, y):
            grads = jax.grad(loss_fn)(model, x, y)
            return optimizer.update(model, grads)

        initial_loss = loss_fn(model, x, y)
        for _ in range(10):
            model, optimizer = step(model, optimizer, x, y)

        assert loss_fn(model, x, y) < initial_loss
        assert optimizer.step == 10

    def test_lax_scan_compatible(self):
        """Training loop via jax.lax.scan with Optimizer in carry."""
        model = nn.Linear(4, 1, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.sgd(0.01), model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        def scan_step(carry, _):
            model, optimizer = carry
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            model, optimizer = optimizer.update(model, grads)
            return (model, optimizer), loss

        initial_loss = loss_fn(model, x, y)
        (model, optimizer), losses = jax.lax.scan(
            scan_step,
            (model, optimizer),
            None,
            length=20,
        )
        assert loss_fn(model, x, y) < initial_loss
        assert losses[-1] < losses[0]
        assert optimizer.step == 20


class TestOptimizerAutoPartition:
    def test_lora_training_no_manual_partition(self):
        """Frozen base unchanged, loss decreases, no manual partition boilerplate."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(4, 4, key=keys[0])
        lora = nn.LoRALinear(linear, rank=2, key=keys[1])
        frozen_base_w = lora.linear.w._value.copy()

        optimizer = ion.Optimizer(optax.adam(1e-2), lora)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 4))

        def loss_fn(model, x, y):
            return jnp.mean((model(x) - y) ** 2)

        initial_loss = loss_fn(lora, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(lora, x, y)
            lora, optimizer = optimizer.update(lora, grads)

        assert loss_fn(lora, x, y) < initial_loss
        npt.assert_array_equal(lora.linear.w._value, frozen_base_w)

    def test_frozen_params_no_wasted_memory(self):
        """set_to_zero() allocates fewer state leaves than adam."""
        model = nn.Linear(4, 4, key=jax.random.key(0))
        frozen = model.freeze()

        opt_trainable = ion.Optimizer(optax.adam(1e-3), model)
        opt_frozen = ion.Optimizer(optax.adam(1e-3), frozen)

        leaves_trainable = jax.tree.leaves(opt_trainable.state)
        leaves_frozen = jax.tree.leaves(opt_frozen.state)
        assert len(leaves_frozen) < len(leaves_trainable)

    def test_mixed_trainable_frozen(self):
        """Trainable params update, frozen params unchanged."""

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

        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m.decoder(m.encoder(x)) - y) ** 2)

        initial_loss = loss_fn(model, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(model, x, y)
            model, optimizer = optimizer.update(model, grads)

        assert loss_fn(model, x, y) < initial_loss
        npt.assert_array_equal(model.encoder.w._value, frozen_w)

    def test_mixed_trainable_frozen_and_bare_array(self):
        """Trainable, frozen, and non-Param leaves all handled correctly."""

        class MixedModel(nn.Module):
            trainable_layer: nn.Linear
            frozen_layer: nn.Linear
            scale: jax.Array

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.trainable_layer = nn.Linear(4, 4, key=keys[0])
                self.frozen_layer = nn.Linear(4, 1, key=keys[1]).freeze()
                self.scale = jnp.array(1.0)

        model = MixedModel(key=jax.random.key(0))
        frozen_w = model.frozen_layer.w._value.copy()
        scale_before = model.scale.copy()

        optimizer = ion.Optimizer(optax.adam(1e-2), model)

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 1))

        def loss_fn(m, x, y):
            return jnp.mean((m.frozen_layer(m.trainable_layer(x)) * m.scale - y) ** 2)

        initial_loss = loss_fn(model, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(model, x, y)
            model, optimizer = optimizer.update(model, grads)

        assert loss_fn(model, x, y) < initial_loss
        npt.assert_array_equal(model.frozen_layer.w._value, frozen_w)
        npt.assert_array_equal(model.scale, scale_before)

    def test_bare_array_no_optimizer_state(self):
        """Non-Param arrays don't get optimizer state even when all Params are trainable."""
        pure = nn.Linear(4, 4, key=jax.random.key(0))
        mixed = WithBareArray(key=jax.random.key(0))

        opt_pure = ion.Optimizer(optax.adam(1e-3), pure)
        opt_mixed = ion.Optimizer(optax.adam(1e-3), mixed)

        # WithBareArray.buf should not contribute optimizer state.
        # pure Linear has 2 Params (w, b); WithBareArray has 1 Param (w) + 1 bare array (buf).
        # So mixed should have fewer state leaves despite the same total leaf count.
        # NOTE: This comparison relies on adam allocating >1 state leaf per param (mu + nu).
        # If optax changes its internal state structure, the leaf counts may shift but the
        # directional invariant (fewer params -> fewer state leaves) should still hold.
        assert len(jax.tree.leaves(opt_mixed.state)) < len(jax.tree.leaves(opt_pure.state))


class TestOptimizerCheckpoint:
    def test_save_load_roundtrip(self):
        """ion.save/ion.load preserves optimizer state and step."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)

        # Advance one step so step > 0
        grads = jax.grad(lambda m: jnp.sum(m(jnp.ones((1, 4)))))(model)
        model, optimizer = optimizer.update(model, grads)

        import os
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "opt_test.npz")
        try:
            ion.save(path, optimizer)
            loaded = ion.load(path, optimizer)
            assert loaded.step == optimizer.step
            for a, b in zip(jax.tree.leaves(loaded.state), jax.tree.leaves(optimizer.state)):
                npt.assert_array_equal(a, b)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestOptimizerStructureMismatch:
    """Mismatched pytree structures should produce clear errors, not silent corruption."""

    def test_different_model_structure_raises(self):
        """Optimizer init'd on model A rejects model B with a different pytree structure."""

        class SingleLayer(nn.Module):
            layer: nn.Linear

            def __init__(self, key):
                self.layer = nn.Linear(4, 2, key=key)

            def __call__(self, x):
                return self.layer(x)

        class TwoLayer(nn.Module):
            layer1: nn.Linear
            layer2: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key)
                self.layer1 = nn.Linear(4, 4, key=keys[0])
                self.layer2 = nn.Linear(4, 2, key=keys[1])

            def __call__(self, x):
                return self.layer2(self.layer1(x))

        model_a = SingleLayer(key=jax.random.key(0))
        model_b = TwoLayer(key=jax.random.key(1))

        optimizer = ion.Optimizer(optax.adam(1e-3), model_a)
        grads_b = jax.grad(lambda m: jnp.mean(m(jnp.ones((2, 4))) ** 2))(model_b)

        with pytest.raises(Exception):
            optimizer.update(model_b, grads_b)

    def test_different_model_type_raises(self):
        """Optimizer init'd on Linear rejects WithBareArray (different leaf count)."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)

        other = WithBareArray(key=jax.random.key(1))
        grads_other = jax.grad(lambda m: jnp.sum(m(jnp.ones(4))))(other)

        with pytest.raises(Exception):
            optimizer.update(other, grads_other)


class TestOptimizerRepr:
    def test_repr_includes_step_and_leaves(self):
        """Repr string includes step count and state leaf count."""
        model = nn.Linear(4, 2, key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        r = repr(optimizer)
        assert "step=0" in r
        assert "state_leaves=" in r
