import dataclasses

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

import ion
from ion import display, nn
from ion.cost import Cost, LayerCost


def analysis(model, *args, **kwargs):
    """Return XLA's FLOP total for the same call."""
    compiled = jax.jit(lambda m, *a: m(*a, **kwargs)).lower(model, *args).compile()
    reported = compiled.cost_analysis()
    assert reported is not None
    reported = reported[0] if isinstance(reported, list) else reported
    return reported.get("flops", 0.0)


class _TwoHead(nn.Module):
    """Two heads over a shared torso, so each method reaches a different branch."""

    torso: nn.MLP
    policy: nn.Linear
    value: nn.Linear

    def __init__(self, *, key):
        keys = jax.random.split(key, 3)
        self.torso = nn.MLP([16, 32], key=keys[0])
        self.policy = nn.Linear(32, 4, key=keys[1])
        self.value = nn.Linear(32, 1, key=keys[2])

    def actor(self, x):
        return self.policy(self.torso(x))

    def critic(self, x):
        return self.value(self.torso(x))

    def __call__(self, x):
        return self.actor(x), self.critic(x)


class TestTotals:
    @pytest.mark.parametrize(
        "model, shape",
        [
            (nn.MLP([256, 512, 512, 10], key=jax.random.key(0)), (32, 256)),
            (nn.Linear(1024, 1024, key=jax.random.key(0)), (128, 1024)),
            (nn.Conv(3, 64, (3, 3), key=jax.random.key(0)), (16, 32, 32, 3)),
            (nn.MultiHeadAttention(384, num_heads=6, key=jax.random.key(0)), (8, 128, 384)),
        ],
    )
    def test_flops_match_xla(self, model, shape):
        """FLOPs agree with XLA on calls without static loops."""
        measured = ion.cost(model, jnp.ones(shape))
        npt.assert_allclose(measured.flops, analysis(model, jnp.ones(shape)), rtol=0.01)

    def test_elementwise_flops_are_indicative(self):
        """The conventional elementwise count stays reasonably close to XLA's."""
        model = nn.LayerNorm(384)
        measured = ion.cost(model, jnp.ones((8, 128, 384)))
        npt.assert_allclose(measured.flops, analysis(model, jnp.ones((8, 128, 384))), rtol=0.5)

    def test_params_and_ops_are_populated(self):
        """The headline exposes model size and graph simplification."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, 64)))
        assert measured.params == model.num_params
        assert measured.param_bytes == sum(x.nbytes for x in jax.tree.leaves(model.params))
        assert measured.ops > 0 and measured.fused > 0
        assert measured.fused < measured.ops


class TestMemory:
    def test_total_is_the_compiler_memory_equation(self):
        """Inputs, intermediate and outputs reconcile with reused buffers removed."""
        measured = ion.cost(nn.MLP([64, 128, 10], key=jax.random.key(0)), jnp.ones((8, 64)))
        assert measured.total_memory == (
            measured.input_bytes
            + measured.intermediate_bytes
            + measured.output_bytes
            - measured.reused_bytes
        )

    def test_input_splits_data_and_parameters(self):
        """The common model call shows data followed by parameter storage."""

        model = nn.Linear(8, 4, key=jax.random.key(0))
        x = jnp.ones((2, 8))
        measured = ion.cost(model, x)
        assert measured.input_bytes == x.nbytes + measured.param_bytes
        lines = repr(measured).splitlines()
        params = next(line for line in lines if line.startswith("params"))
        inputs = [line for line in lines if line.startswith("input ")][-1]
        assert display.scaled(measured.param_bytes) in params
        assert display.scaled(x.nbytes) in inputs

    def test_memory_grows_with_the_batch(self):
        """The compiler memory plan reflects larger call inputs and outputs."""
        model = nn.MLP([256, 512, 10], key=jax.random.key(0))
        small = ion.cost(model, jnp.ones((32, 256))).total_memory
        large = ion.cost(model, jnp.ones((256, 256))).total_memory
        assert large > small


class TestScan:
    @pytest.mark.parametrize("length", [1, 4, 16, 64])
    def test_flops_scale_with_sequence(self, length):
        """A scan body runs once in the graph but its arithmetic runs at every step."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        measured = ion.cost(model, jnp.ones((8, length, 64)))
        matmuls = (2 * 8 * 64 * 384 + 2 * 8 * 128 * 384) * length
        npt.assert_allclose(measured.flops, matmuls, rtol=0.03)

    def test_records_loop_count(self):
        """The table can show the static multiplier without inflating graph ops."""
        model = nn.GRU(64, 128, key=jax.random.key(0))
        brief = ion.cost(model, jnp.ones((8, 1, 64)))
        long = ion.cost(model, jnp.ones((8, 32, 64)))
        assert long.layers[""].loop == 32
        assert long.ops == brief.ops


class TestStructure:
    def test_mirrors_the_module_tree(self):
        """Paths, labels and depths match the tree printed by the module repr."""
        model = nn.Sequential(nn.Linear(8, 16, key=jax.random.key(0)), nn.LayerNorm(16))
        layers = ion.cost(model, jnp.ones((4, 8))).layers
        assert list(layers) == ["", "layers[0]", "layers[1]"]
        assert [layer.name for layer in layers.values()] == ["Sequential", "Linear", "LayerNorm"]
        assert [layer.label for layer in layers.values()] == ["", "(0)", "(1)"]
        assert [layer.depth for layer in layers.values()] == [0, 1, 1]

    def test_nested_paths(self):
        """Nested scopes rebuild the full dotted path of every descendant."""
        inner = nn.Sequential(nn.Linear(8, 8, key=jax.random.key(0)))
        layers = ion.cost(nn.Residual(inner), jnp.ones((4, 8))).layers
        assert list(layers) == ["", "layer", "layer.layers[0]"]

    def test_outputs_are_recorded_from_the_trace(self):
        """Each module carries its logical output shape and dtype."""
        model = nn.Sequential(
            nn.Linear(8, 16, key=jax.random.key(0)),
            nn.Linear(16, 4, key=jax.random.key(1)),
        )
        layers = ion.cost(model, jnp.ones((2, 8))).layers
        assert layers["layers[0]"].output == jax.ShapeDtypeStruct((2, 16), jnp.float32)
        assert layers["layers[1]"].output == jax.ShapeDtypeStruct((2, 4), jnp.float32)

    def test_mapped_outputs_carry_the_axis_vmap_adds(self):
        """A batch tracer hides the mapped axis, so outputs report the shape XLA materializes."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        x = jnp.ones((32, 64))
        call = lambda m, z: jax.vmap(lambda module, row: module(row), in_axes=(None, 0))(m, z)
        mapped = ion.cost(call, model, x)
        direct = ion.cost(model, x)
        assert [layer.output for layer in mapped.layers.values()] == [
            layer.output for layer in direct.layers.values()
        ]

    def test_mapped_outputs_compose_with_other_transforms(self):
        """Nesting and reverse mode stack their axes, so every mapped axis is restored."""
        model = nn.MLP([8, 16, 4], key=jax.random.key(0))
        row = lambda module, value: module(value)
        nested = ion.cost(
            lambda m, z: jax.vmap(jax.vmap(row, in_axes=(None, 0)), in_axes=(None, 0))(m, z),
            model,
            jnp.ones((3, 5, 8)),
        )
        assert nested.layers["layers[0]"].output.shape == (3, 5, 16)

        gradient = ion.cost(
            lambda m, z: jax.vmap(jax.grad(lambda module, value: jnp.sum(module(value))))(m, z),
            jax.tree.map(lambda *leaves: jnp.stack(leaves), model, model),
            jnp.ones((2, 8)),
        )
        assert gradient.layers["layers[0]"].output.shape == (2, 16)

    def test_structured_output_keeps_its_pytree(self):
        """Multiple outputs retain their original structure."""

        class Split(nn.Module):
            def __init__(self):
                pass

            def __call__(self, x):
                return {"values": x * 2, "mean": x.mean()}

        output = ion.cost(Split(), jnp.ones((2, 8))).layers[""].output
        assert output == {
            "mean": jax.ShapeDtypeStruct((), jnp.float32),
            "values": jax.ShapeDtypeStruct((2, 8), jnp.float32),
        }


class TestShareAndOps:
    def test_share_is_a_fraction_of_total_flops(self):
        """Share now has the direct and device-independent FLOP meaning."""
        measured = ion.cost(nn.MLP([256, 512, 512, 10], key=jax.random.key(0)), jnp.ones((64, 256)))
        for layer in measured.layers.values():
            npt.assert_allclose(layer.share, layer.flops / measured.flops)

    def test_children_fit_within_parent(self):
        """Inclusive child work leaves any direct parent work as an unfilled bar segment."""
        layers = ion.cost(
            nn.MLP([64, 128, 256, 10], key=jax.random.key(0)), jnp.ones((8, 64))
        ).layers
        assert sum(layer.flops for path, layer in layers.items() if path) <= layers[""].flops
        assert sum(layer.ops for path, layer in layers.items() if path) <= layers[""].ops

    def test_parent_totals_are_inclusive(self):
        """The root owns the complete call while descendants own their subtrees."""
        measured = ion.cost(nn.Residual(nn.Linear(8, 8, key=jax.random.key(0))), jnp.ones((2, 8)))
        assert measured.layers[""].flops == measured.flops
        assert measured.layers[""].ops == measured.ops
        assert measured.layers["layer"].flops < measured.flops


class TestTargets:
    def test_module(self):
        """A module is called directly with the arguments that follow it."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        assert ion.cost(model, jnp.ones((8, 64))).flops > 0

    def test_module_method(self):
        """`model.cost(x)` matches the function called on the same model."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        x = jnp.ones((8, 64))
        method, function = model.cost(x), ion.cost(model, x)
        assert method.flops == function.flops
        assert method.layers.keys() == function.layers.keys()

    def test_loss_function(self):
        """A function may receive the model among its arguments."""
        model = nn.MLP([64, 32, 10], key=jax.random.key(0))
        loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
        measured = ion.cost(loss, model, jnp.ones((8, 64)), jnp.ones((8, 10)))
        assert measured.flops > 0

    def test_keyword_model(self):
        """The model may also be supplied by keyword."""
        model = nn.Linear(8, 4, key=jax.random.key(0))
        measured = ion.cost(lambda x, model: model(x), jnp.ones((2, 8)), model=model)
        assert measured.layers[""].output.shape == (2, 4)

    def test_gradient_breaks_down_by_layer(self):
        """Reverse-mode work is charged to the layer whose forward pass produced it."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
        x, y = jnp.ones((8, 64)), jnp.ones((8, 10))
        forward = ion.cost(model, x)
        gradient = ion.cost(jax.grad(loss), model, x, y)
        assert gradient.flops > 1.5 * forward.flops
        assert list(gradient.layers) == list(forward.layers)
        assert all(gradient.layers[p].flops > forward.layers[p].flops for p in forward.layers)

    @pytest.mark.parametrize(
        "transform",
        [
            lambda call: jax.jit(call),
            lambda call: jax.checkpoint(call),
            lambda call: lambda m, x: jax.vmap(call, in_axes=(None, 0))(m, x),
        ],
    )
    def test_transforms_that_rebuild_the_tree_keep_their_layers(self, transform):
        """Flattening a model into a transform rebuilds it, and the copy reclaims its paths."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        x = jnp.ones((8, 64))
        direct = ion.cost(model, x)
        assert list(ion.cost(transform(lambda m, x: m(x)), model, x).layers) == list(direct.layers)

    def test_bound_method_target(self):
        """A bound method names both its model and the call, so it stands alone as the target."""
        model = _TwoHead(key=jax.random.key(0))
        critic = ion.cost(model.critic, jnp.ones((8, 16)))
        assert list(critic.layers) == ["", "torso", "torso.layers[0]", "value"]
        assert critic.layers[""].output == jax.ShapeDtypeStruct((8, 1), jnp.float32)
        assert critic.name == "_TwoHead.critic"

    def test_named_method_on_the_model(self):
        """The method interface names a call the model does not expose as its forward pass."""
        model = _TwoHead(key=jax.random.key(0))
        x = jnp.ones((8, 16))
        named = model.cost(x, method="critic")
        assert named.flops == ion.cost(model.critic, x).flops
        assert list(named.layers) == ["", "torso", "torso.layers[0]", "value"]
        assert named.name == "_TwoHead.critic"

    def test_rejects_an_unknown_method(self):
        """A method the model does not define fails on attribute access."""
        model = _TwoHead(key=jax.random.key(0))
        with pytest.raises(AttributeError, match="critc"):
            model.cost(jnp.ones((8, 16)), method="critc")

    def test_forward_pass_is_the_default_call(self):
        """Naming the forward pass explicitly matches passing the model itself."""
        model = _TwoHead(key=jax.random.key(0))
        x = jnp.ones((8, 16))
        assert ion.cost(model.__call__, x).flops == ion.cost(model, x).flops
        assert ion.cost(model, x).name == "_TwoHead"

    def test_sibling_methods_measure_their_own_branch(self):
        """Each method reports only the layers its own call reaches."""
        model = _TwoHead(key=jax.random.key(0))
        x = jnp.ones((8, 16))
        assert "policy" in ion.cost(model.actor, x).layers
        assert "value" not in ion.cost(model.actor, x).layers
        assert "policy" not in ion.cost(model.critic, x).layers
        assert ion.cost(model, x).flops > ion.cost(model.critic, x).flops

    def test_report_renders_with_an_optimizer_argument(self):
        """A training step carries optimizer state, whose abstract step has no concrete value."""
        model = nn.MLP([16, 8], key=jax.random.key(0))
        optimizer = ion.Optimizer(optax.adam(1e-3), model)
        x, y = jnp.ones((4, 16)), jnp.ones((4, 8))

        def step(m, opt, inputs, targets):
            loss, grads = jax.value_and_grad(lambda t: jnp.mean((t(inputs) - targets) ** 2))(m)
            m, opt = opt.update(m, grads)
            return m, opt, loss

        assert "step=" in repr(ion.cost(step, model, optimizer, x, y))

    def test_static_arguments(self):
        """Non-array positional and keyword configuration is compiled statically."""
        model = nn.Linear(8, 4, key=jax.random.key(0))
        call = lambda m, x, mode, *, enabled: m(x) if mode == "train" and enabled else x[:, :4]
        assert ion.cost(call, model, jnp.ones((2, 8)), "train", enabled=True).flops > 0

    def test_nested_concrete_inputs_are_abstractified(self):
        """Array pytrees and explicit shape pytrees produce the same analysis."""

        class Add(nn.Module):
            def __init__(self):
                pass

            def __call__(self, batch):
                return batch["x"] + batch["y"]

        concrete = {"x": jnp.ones((2, 8)), "y": jnp.ones((2, 8))}
        shaped = {
            key: jax.ShapeDtypeStruct(value.shape, value.dtype) for key, value in concrete.items()
        }
        assert ion.cost(Add(), concrete).flops == ion.cost(Add(), shaped).flops

    def test_rejects_a_call_without_a_module(self):
        """There is no module tree to report without a Module in the call."""
        with pytest.raises(TypeError, match="needs a Module"):
            ion.cost(lambda x: x * 2, jnp.ones((8,)))

    def test_rejects_dynamic_control_flow(self):
        """Unknown runtime paths are rejected instead of reported as a false total."""

        class Branch(nn.Module):
            def __init__(self):
                pass

            def __call__(self, x, pred):
                return jax.lax.cond(pred, lambda y: y * 2, lambda y: y + 1, x)

        with pytest.raises(NotImplementedError, match="dynamic cond"):
            ion.cost(Branch(), jnp.ones((8,)), jnp.array(True))


class TestReport:
    def test_is_frozen_and_not_a_pytree(self):
        """A report is immutable metadata rather than model data."""
        measured = ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert dataclasses.is_dataclass(Cost) and dataclasses.is_dataclass(LayerCost)
        assert jax.tree.leaves(measured) == [measured]
        with pytest.raises(dataclasses.FrozenInstanceError):
            measured.flops = 0  # pyright: ignore[reportAttributeAccessIssue]

    def test_repr_contains_each_section(self):
        """The report prints totals, memory composition and the compact layer table."""
        model = nn.Sequential(nn.Linear(8, 16, key=jax.random.key(0)), nn.LayerNorm(16))
        text = repr(ion.cost(model, jnp.ones((4, 8))))
        assert "memory\ntotal" in text and "\nintermediate" in text
        assert " params · " in text and "FLOP · " in text
        assert "ops" in text and "fused" in text and "output" in text
        assert "f32(4, 16)" in text

    def test_reused_memory_is_conditional_and_subtractive(self, monkeypatch):
        """Aliased memory appears only when present and uses a dotted subtraction bar."""
        monkeypatch.setenv("NO_COLOR", "1")
        measured = ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert "\nreused" not in repr(measured)

        reused = dataclasses.replace(measured, reused_bytes=measured.output_bytes)
        line = next(line for line in repr(reused).splitlines() if line.startswith("reused"))
        assert f"-{display.scaled(reused.reused_bytes)}" in line

    def test_forward_passes_are_restored(self):
        """The scope wrapper is installed for the trace alone and removed afterwards."""

        original = nn.Linear.__call__
        ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert nn.Linear.__call__ is original
