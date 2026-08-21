import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy.testing as npt
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
        described = f"({display.scaled(x.nbytes)} + {display.scaled(measured.param_bytes)}) input"
        assert described in repr(measured)

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

    def test_gradient_reports_the_call_as_a_whole(self):
        """A transform rebuilds the tree, so reverse-mode work is totalled without a breakdown."""
        model = nn.MLP([64, 128, 10], key=jax.random.key(0))
        loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
        x, y = jnp.ones((8, 64)), jnp.ones((8, 10))
        forward = ion.cost(model, x)
        gradient = ion.cost(jax.grad(loss), model, x, y)
        assert gradient.flops > 1.5 * forward.flops
        assert list(gradient.layers) == [""] and list(forward.layers) != [""]

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

    def test_repr_contains_the_new_contract(self):
        """The report prints totals, memory composition and the compact layer table."""
        model = nn.Sequential(nn.Linear(8, 16, key=jax.random.key(0)), nn.LayerNorm(16))
        text = repr(ion.cost(model, jnp.ones((4, 8))))
        assert "total memory =" in text and " intermediate" in text
        assert "ops" in text and "fused" in text and "output" in text
        assert "float32(4, 16)" in text
        assert "ceiling" not in text and "transfer" not in text

    def test_repr_colors_dtypes_alone(self, monkeypatch):
        """Only dtypes are blue, so a row of figures reads as one measurement."""

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        measured = ion.cost(nn.Linear(8, 16, key=jax.random.key(0)), jnp.ones((4, 8)))
        text = repr(measured)
        shape = f"{display._SYMBOL}float32\x1b[0m(4, 16)"
        op_width = max(len("ops"), len(f"{measured.ops:,}"))

        assert shape in text
        assert f"{measured.ops:>{op_width},}  {shape}" in text
        assert f"{display.scaled(measured.flops, 1e3, display._FLOPS):>7}" in text
        assert " 100.0%" in text
        assert display._NUMBER not in text

    def test_sibling_bars_do_not_share_a_character_cell(self, monkeypatch):
        """A sibling starts after the cell occupied by its predecessor's partial block."""
        monkeypatch.setenv("NO_COLOR", "1")
        measured = ion.cost(nn.MLP([8, 17, 4], key=jax.random.key(0)), jnp.ones((4, 8)))
        lines = repr(measured).splitlines()
        header = next(line for line in lines if line.startswith("layer"))
        bar_start = header.index("share") - 12
        first = next(line for line in lines if "(0) Linear" in line)[bar_start : bar_start + 10]
        second = next(line for line in lines if "(1) Linear" in line)[bar_start : bar_start + 10]
        first_cells = {index for index, cell in enumerate(first) if not cell.isspace()}
        second_cells = {index for index, cell in enumerate(second) if not cell.isspace()}

        assert first_cells and second_cells
        assert max(first_cells) < min(second_cells)

    def test_forward_passes_are_restored(self):
        """The scope wrapper is installed for the trace alone and removed afterwards."""

        original = nn.Linear.__call__
        ion.cost(nn.Linear(8, 4, key=jax.random.key(0)), jnp.ones((2, 8)))
        assert nn.Linear.__call__ is original
