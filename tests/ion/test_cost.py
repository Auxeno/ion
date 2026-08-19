import dataclasses

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import ion
from ion import nn
from ion.cost import Cost, LayerCost


def analysis(model, *args, **kwargs):
    """Return XLA's own totals for the same call, as the reference to check against."""
    compiled = jax.jit(lambda m, *a: m(*a, **kwargs)).lower(model, *args).compile()
    reported = compiled.cost_analysis()
    reported = reported[0] if isinstance(reported, list) else reported
    return reported.get("flops", 0.0), reported.get("bytes accessed", 0.0)


class TestTotals:
    @pytest.mark.parametrize(
        "model, shape",
        [
            (nn.MLP([256, 512, 512, 10], key=jax.random.key(0)), (32, 256)),
            (nn.MLP([256] * 8, key=jax.random.key(0)), (128, 256)),
            (nn.Linear(1024, 1024, key=jax.random.key(0)), (128, 1024)),
            (nn.Conv(3, 64, (3, 3), key=jax.random.key(0)), (16, 32, 32, 3)),
            (nn.MultiHeadAttention(384, num_heads=6, key=jax.random.key(0)), (8, 128, 384)),
        ],
    )
    def test_matches_xla(self, model, shape):
        """FLOPs and traffic agree with XLA's own analysis on models without a scan."""
        measured = ion.cost(model, jnp.ones(shape))
        flops, transferred = analysis(model, jnp.ones(shape))
        npt.assert_allclose(measured.flops, flops, rtol=0.01)
        npt.assert_allclose(measured.transferred, transferred, rtol=0.001)

    def test_elementwise_flops_are_indicative(self):
        """Counting an elementwise chain by its outputs approximates what XLA charges for it."""
        measured = ion.cost(nn.LayerNorm(384), jnp.ones((8, 128, 384)))
        flops, transferred = analysis(nn.LayerNorm(384), jnp.ones((8, 128, 384)))
        npt.assert_allclose(measured.transferred, transferred, rtol=0.001)
        npt.assert_allclose(measured.flops, flops, rtol=0.5)

    def test_embedding_traffic_dominates(self):
        """A lookup moves far more than it computes, so its intensity stays near zero."""
        model = nn.Embedding(16384, 384, key=jax.random.key(0))
        measured = ion.cost(model, jnp.zeros((8, 128), jnp.int32))
        assert measured.layers[""].intensity < 1.0
        assert measured.layers[""].ceiling < 0.001

    def test_totals_are_positive(self):
        """Every headline figure is populated for an ordinary forward pass."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, 64)))
        assert measured.flops > 0 and measured.transferred > 0
        assert measured.peak_memory > 0 and measured.memory > 0
        assert measured.ops > 0 and measured.kernels > 0
        assert measured.params == model.num_params


class TestScan:
    @pytest.mark.parametrize("length", [1, 4, 16, 64])
    def test_scales_with_sequence(self, length):
        """A scan body runs once in the jaxpr, so its cost is scaled by the sequence length."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, length, 64)))

        # Three gates over both the input and the hidden state, per step
        matmuls = (2 * 8 * 64 * 384 + 2 * 8 * 128 * 384) * length
        npt.assert_allclose(measured.flops, matmuls, rtol=0.03)

    def test_records_loop_count(self):
        """The layer carries the count its body was scaled by, so the table can show it."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        assert ion.cost(model, jnp.ones((8, 32, 64))).layers[""].loop == 32

    def test_cost_analysis_undercounts(self):
        """XLA counts a scan body once, which is the reason this analysis exists."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, 64, 64)))
        assert measured.flops > 50 * analysis(model, jnp.ones((8, 64, 64)))[0]


class TestStructure:
    def test_mirrors_the_module_tree(self):
        """Paths, labels and depths match the tree the module repr prints."""
        model = nn.Sequential(nn.Linear(8, 16, key=jax.random.key(0)), nn.LayerNorm(16))
        layers = ion.cost(model, jnp.ones((4, 8))).layers
        assert list(layers) == ["", "layers[0]", "layers[1]"]
        assert [layer.name for layer in layers.values()] == ["Sequential", "Linear", "LayerNorm"]
        assert [layer.label for layer in layers.values()] == ["", "(0)", "(1)"]
        assert [layer.depth for layer in layers.values()] == [0, 1, 2 - 1]

    def test_nested_paths(self):
        """Nested scopes rebuild the full dotted path of every descendant."""
        inner = nn.Sequential(nn.Linear(8, 8, key=jax.random.key(0)))
        layers = ion.cost(nn.Residual(inner), jnp.ones((4, 8))).layers
        assert list(layers) == ["", "layer", "layer.layers[0]"]

    def test_root_holds_the_whole_call(self):
        """The root row totals the model, so its share fills the bar."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, 64)))
        npt.assert_allclose(measured.layers[""].share, 1.0)
        assert measured.layers[""].flops == measured.flops
        assert measured.layers[""].transferred == measured.transferred

    def test_children_sum_within_parent(self):
        """A parent's arithmetic is the sum of what its children were charged."""
        model = nn.MLP([64, 128, 256, 10], key=jax.random.key(0))
        layers = ion.cost(model, jnp.ones((8, 64))).layers
        children = sum(layer.flops for path, layer in layers.items() if path)
        assert children <= layers[""].flops


class TestMemory:
    def test_tracks_the_compiler(self):
        """Peak live bytes land near what the compiler reserves when it cannot fuse them away."""
        model = nn.MultiHeadAttention(384, num_heads=6, key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, 128, 384)))
        npt.assert_allclose(measured.memory, measured.peak_memory, rtol=0.1)

    def test_is_a_high_water_mark(self):
        """Buffers are reused as values die, so a deeper model does not hold more at once."""
        shallow = ion.cost(nn.MLP([256] * 4, key=jax.random.key(0)), jnp.ones((128, 256)))
        deep = ion.cost(nn.MLP([256] * 16, key=jax.random.key(0)), jnp.ones((128, 256)))
        assert deep.flops > 3 * shallow.flops
        assert deep.memory == shallow.memory

    def test_grows_with_the_batch(self):
        """Memory is what a larger batch costs, unlike the parameters it flows through."""
        model = nn.MLP([256, 512, 10], key=jax.random.key(0))
        small = ion.cost(model, jnp.ones((32, 256))).memory
        large = ion.cost(model, jnp.ones((256, 256))).memory
        npt.assert_allclose(large / small, 8.0, rtol=0.2)

    def test_a_scan_reuses_its_carry(self):
        """A loop runs its body in place, so only its stacked output grows with the sequence."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        brief = ion.cost(model, jnp.ones((8, 4, 64)))
        long = ion.cost(model, jnp.ones((8, 64, 64)))
        assert long.memory / brief.memory < long.flops / brief.flops


class TestShare:
    def test_is_a_fraction_of_the_whole_call(self):
        """Share is global, so the root holds all of it and no layer exceeds its parent."""
        layers = ion.cost(nn.MLP([256, 512, 512, 10], key=jax.random.key(0)), jnp.ones((64, 256)))
        npt.assert_allclose(layers.layers[""].share, 1.0)
        for path, layer in layers.layers.items():
            assert 0.0 <= layer.share <= 1.0
            if path:
                parent = path.rsplit(".", 1)[0] if "." in path else ""
                assert layer.share <= layers.layers[parent].share + 1e-9

    def test_mixed_bounds_still_divide_their_parent(self):
        """Summing per kernel stops a compute bound layer beside a bandwidth bound one from
        outweighing the parent they share."""
        key = jax.random.key(0)
        model = nn.Sequential(
            nn.Embedding(65536, 512, key=key),
            nn.Linear(512, 4096, key=key),
            nn.Linear(4096, 4096, key=key),
        )
        layers = ion.cost(model, jnp.zeros((4096,), jnp.int32)).layers
        assert layers["layers[0]"].ceiling < 0.01 and layers["layers[2]"].ceiling == 1.0
        npt.assert_allclose(sum(x.share for p, x in layers.items() if p), 1.0, atol=0.01)

    def test_children_divide_their_parent(self):
        """Siblings split what their parent was charged, which is what lets their bars tile it."""
        layers = ion.cost(nn.MLP([256, 512, 512, 10], key=jax.random.key(0)), jnp.ones((64, 256)))
        children = [layer.share for path, layer in layers.layers.items() if path]
        assert sum(children) <= layers.layers[""].share + 1e-9

    def test_a_pass_through_hands_over_everything(self):
        """A wrapper doing no work of its own charges its only child the whole call."""
        model = nn.Sequential(nn.Linear(256, 256, key=jax.random.key(0)))
        npt.assert_allclose(ion.cost(model, jnp.ones((64, 256))).layers["layers[0]"].share, 1.0)

    def test_a_wrapper_keeps_its_own_work(self):
        """A residual add is traffic the parent pays for, so the child falls short of the bar."""
        model = nn.Residual(nn.Linear(256, 256, key=jax.random.key(0)))
        assert 0.5 < ion.cost(model, jnp.ones((64, 256))).layers["layer"].share < 0.9

    def test_nesting_does_not_shrink_it(self):
        """Wrappers that only forward a call keep all of it, however many there are."""
        key = jax.random.key(0)
        bare = nn.Sequential(nn.Linear(256, 256, key=key))
        wrapped = nn.Sequential(nn.Sequential(nn.Sequential(nn.Linear(256, 256, key=key))))
        shallow = ion.cost(bare, jnp.ones((64, 256))).layers["layers[0]"].share
        deep = ion.cost(wrapped, jnp.ones((64, 256))).layers["layers[0].layers[0].layers[0]"].share
        npt.assert_allclose(deep, shallow, rtol=0.05)


class TestRoofline:
    def test_ceiling_is_a_fraction(self):
        """Every layer's ceiling sits between nothing and the whole device."""
        model = nn.MLP([256, 512, 10], key=jax.random.key(0))
        for layer in ion.cost(model, jnp.ones((64, 256))).layers.values():
            assert 0.0 <= layer.ceiling <= 1.0

    def test_balance_moves_the_ridge(self):
        """A lower balance makes the same layer look closer to compute limited."""
        model = nn.Linear(1024, 1024, key=jax.random.key(0))
        strict = ion.cost(model, jnp.ones((256, 1024)), balance=500.0)
        lenient = ion.cost(model, jnp.ones((256, 1024)), balance=50.0)
        assert lenient.layers[""].ceiling > strict.layers[""].ceiling

    def test_compute_bound_matmul_reaches_the_top(self):
        """A large square matmul sits past the ridge, so nothing but arithmetic limits it."""
        model = nn.Linear(2048, 2048, key=jax.random.key(0))
        assert ion.cost(model, jnp.ones((2048, 2048)), balance=100.0).layers[""].ceiling == 1.0


class TestTargets:
    def test_module(self):
        """A module is called directly with the arguments that follow it."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        assert ion.cost(model, jnp.ones((8, 64))).flops > 0

    def test_loss_function(self):
        """A function is called with its own arguments, the model among them."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
        measured = ion.cost(loss, model, jnp.ones((8, 64)), jnp.ones((8, 10)))
        assert measured.flops > 0

    def test_gradient_costs_more_than_the_forward_pass(self):
        """A backward pass adds the two transposed matmuls to every forward one."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
        forward = ion.cost(model, jnp.ones((8, 64))).flops
        backward = ion.cost(jax.grad(loss), model, jnp.ones((8, 64)), jnp.ones((8, 10))).flops
        assert backward > 1.5 * forward

    def test_static_keyword(self):
        """Keywords that hold no array compile in as static configuration."""
        key = jax.random.key(0)
        measured = ion.cost(nn.Dropout(0.5), jnp.ones((8, 64)), training=True, key=key)
        assert measured.flops > 0

    def test_shape_dtype_struct(self):
        """A shape stands in for an array, so no input has to be allocated."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        shaped = ion.cost(model, jax.ShapeDtypeStruct((8, 64), jnp.float32))
        assert shaped.flops == ion.cost(model, jnp.ones((8, 64))).flops

    def test_rejects_a_call_without_a_module(self):
        """There is nothing to attribute work to without a module in the call."""
        with pytest.raises(TypeError, match="needs a Module"):
            ion.cost(lambda x: x * 2, jnp.ones((8,)))


class TestPrecision:
    def test_reports_the_compute_dtype(self):
        """The dtype shown is the one the layer's heaviest operation ran in."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        assert ion.cost(model, jnp.ones((8, 64))).layers[""].dtype == "float32"

        half = model.astype(jnp.bfloat16)
        assert ion.cost(half, jnp.ones((8, 64), jnp.bfloat16)).layers[""].dtype == "bfloat16"


class TestReport:
    def test_is_frozen(self):
        """A report describes one call and is never edited afterwards."""
        measured = ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert dataclasses.is_dataclass(Cost) and dataclasses.is_dataclass(LayerCost)
        with pytest.raises(dataclasses.FrozenInstanceError):
            measured.flops = 0

    def test_is_not_a_pytree(self):
        """A report is about a model, not model data, so transforms leave it alone."""
        measured = ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert jax.tree.leaves(measured) == [measured]

    def test_repr_lists_every_layer(self):
        """The table prints one row per module, under the two summary lines."""
        model = nn.Sequential(nn.Linear(8, 16, key=jax.random.key(0)), nn.LayerNorm(16))
        text = repr(ion.cost(model, jnp.ones((4, 8))))
        assert "Sequential" in text and "Linear" in text and "LayerNorm" in text
        assert "params" in text and "kernels" in text

    def test_scopes_do_not_leak(self):
        """Labels are installed only while tracing, so ordinary calls stay unnamed."""
        from ion.nn.module import _scopes

        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        ion.cost(model, jnp.ones((8, 64)))
        assert _scopes == {}
