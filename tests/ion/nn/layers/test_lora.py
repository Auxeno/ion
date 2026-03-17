import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax

import ion
from ion import nn


class TestLoRALinear:
    def test_zero_init(self):
        """Output matches base Linear at initialization (LoRA contribution = 0)."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        x = jax.random.normal(jax.random.key(1), (8,))
        npt.assert_allclose(lora(x), linear(x), rtol=1e-5, atol=1e-5)

    def test_frozen_base(self):
        """Base linear weights are frozen."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        assert lora.linear.w.trainable is False
        assert lora.linear.b.trainable is False  # type: ignore[union-attr]

    def test_trainable_lora_params(self):
        """LoRA a and b matrices are trainable."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        assert lora.a.trainable is True
        assert lora.b.trainable is True

    def test_grad_isolation(self):
        """Gradients only flow to a and b, not base weights."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        x = jax.random.normal(jax.random.key(1), (8,))

        grads = jax.grad(lambda model: model(x).sum())(lora)
        # Base weights should have zero gradients (frozen)
        assert lora.linear.b is not None
        assert grads.linear.w is not None
        assert grads.linear.b is not None
        npt.assert_allclose(grads.linear.w._value, jnp.zeros_like(lora.linear.w._value), atol=1e-7)
        npt.assert_allclose(grads.linear.b._value, jnp.zeros_like(lora.linear.b._value), atol=1e-7)
        # LoRA b is zero-initialized, so grad w.r.t. a is zero at init (chain rule through zero b)
        # But grad w.r.t. b should be non-zero (a is randomly initialized)
        assert grads.b is not None
        assert jnp.any(grads.b._value != 0)

    def test_shapes(self):
        """LoRA matrices have correct shapes."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        assert lora.a.shape == (8, 4)
        assert lora.b.shape == (4, 16)

    def test_output_shape(self):
        """Output shape matches base Linear for batched input."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        x = jax.random.normal(jax.random.key(1), (3, 5, 8))
        assert lora(x).shape == (3, 5, 16)

    def test_no_bias(self):
        """Works with bias=False Linear."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, bias=False, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        x = jax.random.normal(jax.random.key(1), (8,))
        npt.assert_allclose(lora(x), linear(x), rtol=1e-5, atol=1e-5)

    def test_alpha_default(self):
        """Alpha defaults to rank when not specified."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, key=keys[1])
        assert lora.alpha == 4.0

    def test_alpha_custom(self):
        """Custom alpha is stored correctly."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(8, 16, key=keys[0])
        lora = nn.LoRALinear(linear, rank=4, alpha=2.0, key=keys[1])
        assert lora.alpha == 2.0


class TestFreezeUnfreeze:
    def test_module_freeze(self):
        """Module.freeze() freezes all parameters."""
        linear = nn.Linear(8, 16, key=jax.random.key(0))
        frozen = linear.freeze()
        assert frozen.w.trainable is False
        assert frozen.b.trainable is False  # type: ignore[union-attr]

    def test_module_unfreeze(self):
        """Module.unfreeze() unfreezes all parameters."""
        linear = nn.Linear(8, 16, key=jax.random.key(0))
        frozen = linear.freeze()
        unfrozen = frozen.unfreeze()
        assert unfrozen.w.trainable is True
        assert unfrozen.b.trainable is True  # type: ignore[union-attr]

    def test_freeze_returns_new_instance(self):
        """freeze() returns a new instance, original is unchanged."""
        linear = nn.Linear(8, 16, key=jax.random.key(0))
        frozen = linear.freeze()
        assert linear.w.trainable is True
        assert frozen.w.trainable is False

    def test_tree_freeze(self):
        """ion.freeze works on arbitrary pytrees."""
        linear = nn.Linear(8, 16, key=jax.random.key(0))
        frozen = ion.freeze(linear)
        assert frozen.w.trainable is False

    def test_tree_unfreeze(self):
        """ion.unfreeze works on arbitrary pytrees."""
        linear = nn.Linear(8, 16, key=jax.random.key(0))
        frozen = ion.freeze(linear)
        unfrozen = ion.unfreeze(frozen)
        assert unfrozen.w.trainable is True


class TestXLAOptimization:
    """XLA dead code elimination for frozen parameter gradients."""

    def test_frozen_base_skips_backward_flops(self):
        """Freezing the base layer eliminates backward FLOPs for it."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(512, 512, key=keys[0])
        x = jnp.ones((32, 512))
        y = jnp.ones((32, 512))

        def compile_step(lora):
            opt = ion.Optimizer(optax.adam(3e-4), lora)

            @jax.jit
            def step(model, opt, x, y):
                grads = jax.grad(lambda m: jnp.mean((m(x) - y) ** 2))(model)
                return opt.update(model, grads)

            return jax.jit(step).lower(lora, opt, x, y).compile()

        frozen = compile_step(nn.LoRALinear(linear, rank=8, key=keys[1]))
        unfrozen = compile_step(nn.LoRALinear(linear, rank=8, key=keys[1]).unfreeze())

        def flops(compiled):
            cost = compiled.cost_analysis()
            return (cost[0] if isinstance(cost, list) else cost)["flops"]

        # stop_gradient on the frozen base means XLA skips its backward pass
        assert flops(frozen) < flops(unfrozen)

    def test_frozen_grads_no_extra_temp_memory(self):
        """XLA DCEs unused zero gradients, no extra temp memory."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(512, 512, key=keys[0])
        lora = nn.LoRALinear(linear, rank=8, key=keys[1])
        x = jnp.ones((32, 512))
        y = jnp.ones((32, 512))

        def loss_fn(m, x, y):
            return jnp.mean((m(x) - y) ** 2)

        # Naive: ion.Optimizer auto-partitions frozen params
        opt_naive = ion.Optimizer(optax.adam(3e-4), lora)

        @jax.jit
        def step_naive(model, opt, x, y):
            grads = jax.grad(loss_fn)(model, x, y)
            return opt.update(model, grads)

        # Head-only: optimizer never sees frozen params at all
        opt_head_raw = optax.adam(3e-4)
        state_head = opt_head_raw.init((lora.a, lora.b))

        @jax.jit
        def step_head(model, state, x, y):
            from ion.optimizer import _apply_updates

            grads = jax.grad(loss_fn)(model, x, y)
            updates, state = opt_head_raw.update((grads.a, grads.b), state)
            return model.replace(
                a=_apply_updates(model.a, updates[0]),  # type: ignore[index]
                b=_apply_updates(model.b, updates[1]),  # type: ignore[index]
            ), state

        naive = jax.jit(step_naive).lower(lora, opt_naive, x, y).compile()
        head_only = jax.jit(step_head).lower(lora, state_head, x, y).compile()

        def temp_bytes(compiled):
            mem = compiled.memory_analysis()
            return (mem[0] if isinstance(mem, list) else mem).temp_size_in_bytes

        # Frozen zero grads flow through naive adam but XLA eliminates them;
        # temp memory should be no worse than never computing them at all
        assert temp_bytes(naive) <= temp_bytes(head_only)

    def test_partition_with_set_to_zero(self):
        """ion.Optimizer auto-partitions frozen params with set_to_zero()."""
        keys = jax.random.split(jax.random.key(0), 2)
        linear = nn.Linear(4, 4, key=keys[0])
        lora = nn.LoRALinear(linear, rank=2, key=keys[1])

        optimizer = ion.Optimizer(optax.adam(3e-4), lora)
        frozen_w = lora.linear.w._value.copy()

        x = jax.random.normal(jax.random.key(1), (8, 4))
        y = jnp.ones((8, 4))

        def loss_fn(model, x, y):
            return jnp.mean((model(x) - y) ** 2)

        initial_loss = loss_fn(lora, x, y)
        for _ in range(10):
            grads = jax.grad(loss_fn)(lora, x, y)
            lora, optimizer = optimizer.update(lora, grads)

        # Loss decreases, frozen base unchanged, LoRA params updated
        assert loss_fn(lora, x, y) < initial_loss
        npt.assert_array_equal(lora.linear.w._value, frozen_w)
        assert lora.a.trainable and lora.b.trainable
