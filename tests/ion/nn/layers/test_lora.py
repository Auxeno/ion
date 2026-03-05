import jax
import numpy.testing as npt

import ion
from ion import nn


class TestLoRALinear:
    def test_zero_init(self):
        """Output matches base Linear at initialization (LoRA contribution = 0)."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        x = jax.random.normal(jax.random.key(1), (8,))
        npt.assert_allclose(lora(x), linear(x), rtol=1e-5, atol=1e-5)

    def test_frozen_base(self):
        """Base linear weights are frozen."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        assert lora.linear.w.trainable is False
        assert lora.linear.b.trainable is False  # type: ignore[union-attr]

    def test_trainable_lora_params(self):
        """LoRA a and b matrices are trainable."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        assert lora.a.trainable is True
        assert lora.b.trainable is True

    def test_grad_isolation(self):
        """Gradients only flow to a and b, not base weights."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        x = jax.random.normal(jax.random.key(1), (8,))

        @ion.grad
        def loss_grad(model, x):
            return model(x).sum()

        grads = loss_grad(lora, x)
        # Base weights should have no gradients (None)
        assert grads.linear.w is None
        assert grads.linear.b is None
        # LoRA params should have gradients
        assert grads.a is not None
        assert grads.b is not None

    def test_shapes(self):
        """LoRA matrices have correct shapes."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        assert lora.a.shape == (8, 4)
        assert lora.b.shape == (4, 16)

    def test_output_shape(self):
        """Output shape matches base Linear for batched input."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        x = jax.random.normal(jax.random.key(1), (3, 5, 8))
        assert lora(x).shape == (3, 5, 16)

    def test_no_bias(self):
        """Works with bias=False Linear."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, bias=False, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        x = jax.random.normal(jax.random.key(1), (8,))
        npt.assert_allclose(lora(x), linear(x), rtol=1e-5, atol=1e-5)

    def test_alpha_default(self):
        """Alpha defaults to rank when not specified."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, key=k2)
        assert lora.alpha == 4.0

    def test_alpha_custom(self):
        """Custom alpha is stored correctly."""
        k1, k2 = jax.random.split(jax.random.key(0))
        linear = nn.Linear(8, 16, key=k1)
        lora = nn.LoRALinear(linear, rank=4, alpha=2.0, key=k2)
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
