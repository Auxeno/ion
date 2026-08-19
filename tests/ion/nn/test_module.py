import copy
import dataclasses
import re
import sys
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import nn, tree


class TestBaseModule:
    def test_call_raises_not_implemented(self):
        """Base Module.__call__ raises NotImplementedError."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        m = Empty()  # type: ignore[reportCallIssue]
        with pytest.raises(NotImplementedError, match="Empty"):
            m(jnp.ones(3))

    def test_iter_yields_field_values(self):
        """Module.__iter__ yields dataclass field values in order."""
        model = nn.Linear(3, 4, key=jax.random.key(0))
        fields = list(model)
        assert len(fields) == 2  # w and b


class TestSubclassTransformation:
    def test_annotations_become_dataclass_fields(self):
        """Annotated fields are recognized by dataclasses.fields()."""

        class Model(nn.Module):
            a: int
            b: float

            def __init__(self, a: int, b: float):
                self.a = a
                self.b = b

        fields = {f.name for f in dataclasses.fields(Model)}  # type: ignore[arg-type]
        assert fields == {"a", "b"}

    def test_custom_init_preserved(self):
        """Subclass with explicit __init__ keeps its constructor logic."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, dim: int, *, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (dim,)))

        m = Model(4, key=jax.random.key(0))
        assert m.w.shape == (4,)

    def test_generated_init(self):
        """Subclass without explicit __init__ gets one from annotations."""

        class Pair(nn.Module):
            x: int
            y: int

        p = Pair(x=1, y=2)  # type: ignore[call-arg]
        assert p.x == 1
        assert p.y == 2

    def test_unannotated_field_raises(self):
        """Assigning an unannotated attribute in __init__ raises AttributeError."""

        class Model(nn.Module):
            x: int

            def __init__(self):
                self.x = 1
                self.scale = 2.0

        with pytest.raises(AttributeError, match="scale"):
            Model()


class TestImmutability:
    def test_setattr_raises_after_init(self):
        """Assigning to a field after construction raises AttributeError."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        with pytest.raises(AttributeError, match="frozen"):
            m.x = 2

    def test_delattr_raises(self):
        """Deleting a field always raises AttributeError."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        with pytest.raises(AttributeError, match="Cannot delete"):
            del m.x

    def test_error_message_contents(self):
        """Error messages include the class name and attribute name."""

        class MyLayer(nn.Module):
            w: int

            def __init__(self, w: int):
                self.w = w

        m = MyLayer(w=1)
        with pytest.raises(AttributeError, match="MyLayer"):
            m.w = 2
        with pytest.raises(AttributeError, match="'w'"):
            m.w = 2


class TestPytreeRegistration:
    def test_flatten_unflatten_roundtrip(self):
        """Flatten then unflatten reconstructs the module exactly."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self, key: jax.Array):
                keys = jax.random.split(key, 2)
                self.w = nn.Param(jax.random.normal(keys[0], (3, 4)))
                self.b = nn.Param(jnp.zeros(4))

        m = Model(key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(m)
        reconstructed = treedef.unflatten(leaves)
        npt.assert_array_equal(reconstructed.w._value, m.w._value)
        npt.assert_array_equal(reconstructed.b._value, m.b._value)

    def test_children_follow_field_order(self):
        """Non-array fields become static aux; leaves are empty but roundtrip preserves values."""

        class Model(nn.Module):
            first: int
            second: int
            third: int

            def __init__(self, first: int, second: int, third: int):
                self.first = first
                self.second = second
                self.third = third

        m = Model(first=1, second=2, third=3)
        leaves = jax.tree.leaves(m)
        assert leaves == []
        # Roundtrip preserves field values
        reconstructed = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert reconstructed.first == 1
        assert reconstructed.second == 2
        assert reconstructed.third == 3

    def test_unflatten_bypasses_init(self):
        """Unflatten works even when __init__ takes different args than stored fields."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (in_dim, out_dim)))

        m = Model(3, 4, key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(m)
        # This would fail if unflatten tried to call __init__(in_dim, out_dim, key)
        reconstructed = treedef.unflatten(leaves)
        npt.assert_array_equal(reconstructed.w._value, m.w._value)

    def test_tree_map(self):
        """jax.tree.map transforms leaves inside a module."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        m = Model()
        doubled = jax.tree.map(lambda x: x * 2, m)
        assert isinstance(doubled, Model)
        npt.assert_array_equal(doubled.w._value, jnp.array([2.0, 4.0]))

    def test_tree_leaves(self):
        """jax.tree.leaves extracts all leaf values."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0]))
                self.b = nn.Param(jnp.array([2.0]))

        m = Model()
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 2
        npt.assert_array_equal(leaves[0], jnp.array([1.0]))
        npt.assert_array_equal(leaves[1], jnp.array([2.0]))


class TestAt:
    def test_set_field(self):
        """Setting one field returns correct values."""

        class Model(nn.Module):
            a: int
            b: int

            def __init__(self, a: int, b: int):
                self.a = a
                self.b = b

        m = Model(a=1, b=2)
        m2 = m.at.b.set(10)
        assert m2.a == 1
        assert m2.b == 10

    def test_chained_sets(self):
        """Several fields update via chained set calls."""

        class Model(nn.Module):
            a: int
            b: int
            c: int

            def __init__(self, a: int, b: int, c: int):
                self.a = a
                self.b = b
                self.c = c

        m = Model(a=1, b=2, c=3)
        m2 = m.at.a.set(10).at.c.set(30)
        assert m2.a == 10
        assert m2.b == 2
        assert m2.c == 30

    def test_deep_set_shares_siblings(self):
        """Setting a nested leaf rebuilds only the spine; untouched subtrees are shared."""

        class Outer(nn.Module):
            inner: nn.Linear
            other: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.inner = nn.Linear(4, 8, key=keys[0])
                self.other = nn.Linear(8, 4, key=keys[1])

        m = Outer(key=jax.random.key(0))
        new_w = nn.Param(jnp.zeros((4, 8)))
        m2 = m.at.inner.w.set(new_w)
        npt.assert_array_equal(m2.inner.w._value, jnp.zeros((4, 8)))
        assert m2.other is m.other
        assert m2.inner.b is m.inner.b

    def test_tuple_index(self):
        """Setting an element inside a tuple field preserves the container type."""

        class Model(nn.Module):
            layers: tuple

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.layers = (nn.Linear(4, 4, key=keys[0]), nn.Linear(4, 4, key=keys[1]))

        m = Model(key=jax.random.key(0))
        new_w = nn.Param(jnp.zeros((4, 4)))
        m2 = m.at.layers[1].w.set(new_w)
        npt.assert_array_equal(m2.layers[1].w._value, jnp.zeros((4, 4)))
        assert m2.layers[0] is m.layers[0]
        assert isinstance(m2.layers, tuple)

    def test_dict_key(self):
        """Setting a value inside a dict field."""

        class Model(nn.Module):
            heads: dict

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.heads = {"a": nn.Linear(4, 4, key=keys[0]), "b": nn.Linear(4, 4, key=keys[1])}

        m = Model(key=jax.random.key(0))
        new_w = nn.Param(jnp.zeros((4, 4)))
        m2 = m.at.heads["b"].w.set(new_w)
        npt.assert_array_equal(m2.heads["b"].w._value, jnp.zeros((4, 4)))
        assert m2.heads["a"] is m.heads["a"]

    def test_namedtuple_field(self):
        """Setting inside a NamedTuple field works by attribute name and by index."""

        class State(NamedTuple):
            h: jax.Array
            c: jax.Array

        class Model(nn.Module):
            state: State

            def __init__(self):
                self.state = State(h=jnp.ones(2), c=jnp.zeros(2))

        m = Model()
        m2 = m.at.state.h.set(jnp.full(2, 5.0))
        npt.assert_array_equal(m2.state.h, jnp.full(2, 5.0))
        assert isinstance(m2.state, State)
        m3 = m.at.state[1].set(jnp.full(2, 7.0))
        npt.assert_array_equal(m3.state.c, jnp.full(2, 7.0))

    def test_returns_new_instance(self):
        """Original module is unchanged after set."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        m2 = m.at.x.set(2)
        assert m.x == 1
        assert m2.x == 2
        assert m is not m2

    def test_result_is_frozen(self):
        """The copy returned by set is also immutable."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        m2 = m.at.x.set(2)
        with pytest.raises(AttributeError, match="frozen"):
            m2.x = 3

    def test_preserves_type(self):
        """set returns the same subclass type."""

        class Child(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Child(x=1)
        m2 = m.at.x.set(2)
        assert type(m2) is Child

    def test_type_step_sets_all_matches(self):
        """A type step applies the rest of the path to every matching node."""

        class Encoder(nn.Module):
            drop: nn.Dropout
            linear: nn.Linear

            def __init__(self, key):
                self.drop = nn.Dropout(0.1)
                self.linear = nn.Linear(4, 4, key=key)

        class Model(nn.Module):
            encoder: Encoder
            drops: tuple

            def __init__(self, key):
                self.encoder = Encoder(key=key)
                self.drops = (nn.Dropout(0.2), nn.Dropout(0.3))

        m = Model(key=jax.random.key(0))
        m2 = m.at[nn.Dropout].p.set(0.5)
        assert m2.encoder.drop.p == 0.5
        assert m2.drops[0].p == 0.5
        assert m2.drops[1].p == 0.5
        assert m2.encoder.linear is m.encoder.linear

    def test_type_step_scoped_to_prefix(self):
        """A type step after a path prefix only touches matches inside that subtree."""

        class Block(nn.Module):
            drop: nn.Dropout

            def __init__(self):
                self.drop = nn.Dropout(0.1)

        class Model(nn.Module):
            encoder: Block
            drop: nn.Dropout

            def __init__(self):
                self.encoder = Block()
                self.drop = nn.Dropout(0.1)

        m = Model()
        m2 = m.at.encoder[nn.Dropout].p.set(0.5)
        assert m2.encoder.drop.p == 0.5
        assert m2.drop is m.drop

    def test_type_step_replaces_whole_node(self):
        """A type step with no further path replaces each matching node wholesale."""

        class Model(nn.Module):
            drops: tuple

            def __init__(self):
                self.drops = (nn.Dropout(0.2), nn.Dropout(0.3))

        m = Model()
        m2 = m.at[nn.Dropout].set(nn.Dropout(0.9))
        assert m2.drops[0].p == 0.9
        assert m2.drops[1].p == 0.9

    def test_type_step_in_dict(self):
        """A type step finds matches inside dict fields; the original is unchanged."""

        class Model(nn.Module):
            heads: dict

            def __init__(self):
                self.heads = {"a": nn.Dropout(0.1), "b": nn.Dropout(0.2)}

        m = Model()
        m2 = m.at[nn.Dropout].p.set(0.5)
        assert m2.heads["a"].p == 0.5
        assert m2.heads["b"].p == 0.5
        assert m.heads["a"].p == 0.1

    def test_chained_type_steps(self):
        """A second type step fans out inside each match of the first."""

        class Block(nn.Module):
            drop: nn.Dropout

            def __init__(self):
                self.drop = nn.Dropout(0.1)

        class Model(nn.Module):
            blocks: tuple

            def __init__(self):
                self.blocks = (Block(), Block())

        m = Model()
        m2 = m.at[Block][nn.Dropout].p.set(0.5)
        assert m2.blocks[0].drop.p == 0.5
        assert m2.blocks[1].drop.p == 0.5

    def test_type_step_matches_root(self):
        """A type step matching the model itself applies the rest of the path to it."""
        drop = nn.Dropout(0.1)
        drop2 = drop.at[nn.Dropout].p.set(0.5)
        assert drop2.p == 0.5

    def test_type_step_no_match_raises(self):
        """A type step matching nothing raises ValueError."""
        linear = nn.Linear(4, 8, key=jax.random.key(0))
        with pytest.raises(ValueError, match="No Dropout found"):
            linear.at[nn.Dropout].p.set(0.5)


class TestParams:
    def test_filters_non_param_leaves(self):
        """Non-Param fields are preserved as static metadata."""

        class Model(nn.Module):
            w: nn.Param
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))
                self.scale = 3.0

        m = Model()
        params = m.params
        assert isinstance(params.w, nn.Param)
        assert params.scale == 3.0

    def test_no_params(self):
        """Module with zero Param fields has no param leaves."""

        class Config(nn.Module):
            size: int
            rate: float

            def __init__(self, size: int, rate: float):
                self.size = size
                self.rate = rate

        m = Config(size=8, rate=0.1)
        leaves = jax.tree.leaves(m.params)
        assert len(leaves) == 0

    def test_nested_module_params(self):
        """Param leaves in nested child modules are included."""
        key = jax.random.key(0)

        class Container(nn.Module):
            layer1: nn.Linear
            layer2: nn.Linear

            def __init__(self, key: jax.Array):
                keys = jax.random.split(key, 2)
                self.layer1 = nn.Linear(4, 8, key=keys[0])
                self.layer2 = nn.Linear(8, 2, key=keys[1])

        m = Container(key=key)
        param_leaves = jax.tree.leaves(m.params)
        # Linear has w (Param) and b (Param), so 2 layers * 2 params = 4
        assert len(param_leaves) == 4
        for leaf in param_leaves:
            assert isinstance(leaf, jnp.ndarray)


class TestNumParams:
    def test_linear(self):
        """Linear(4, 8) has 4*8 + 8 = 40 parameters."""
        model = nn.Linear(4, 8, key=jax.random.key(0))
        assert model.num_params == 40

    def test_no_bias(self):
        """Linear without bias only counts weights."""
        model = nn.Linear(4, 8, use_bias=False, key=jax.random.key(0))
        assert model.num_params == 32

    def test_nested(self):
        """Nested modules sum all Param leaves."""

        class Net(nn.Module):
            a: nn.Linear
            b: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.a = nn.Linear(4, 8, key=keys[0])
                self.b = nn.Linear(8, 2, key=keys[1])

        model = Net(key=jax.random.key(0))
        # a: 4*8+8=40, b: 8*2+2=18
        assert model.num_params == 58

    def test_no_params(self):
        """Module with no Param fields has 0 parameters."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        model = Empty()
        assert model.num_params == 0

    def test_frozen_params_counted(self):
        """internals.md: num_params includes both trainable and frozen params."""
        model = nn.Linear(4, 8, key=jax.random.key(0))
        trainable_count = model.num_params
        frozen_count = model.freeze().num_params
        assert trainable_count == frozen_count


class TestRepr:
    def test_param_field(self):
        """repr contains Param wrapper for Param fields."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.zeros((2, 3)))

        r = repr(Model())
        assert "Param(" in r

    def test_array_field(self):
        """repr contains dtype and shape for plain array fields."""

        class Model(nn.Module):
            x: jax.Array

            def __init__(self):
                self.x = jnp.zeros((3, 4), dtype=jnp.float32)

        assert "x=float32(3, 4)," in repr(Model())

    def test_callable_field(self):
        """repr contains the function __name__ for callable fields."""

        class Model(nn.Module):
            act: Callable

            def __init__(self):
                self.act = jax.nn.relu

        r = repr(Model())
        assert "relu" in r

    def test_nested_modules(self):
        """Nested module tuple indents with open/close brackets."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        class Container(nn.Module):
            layers: tuple

            def __init__(self):
                self.layers = (Empty(), Empty())

        r = repr(Container())
        assert "(" in r  # tuple open bracket
        assert "Empty()" in r

    def test_empty_module(self):
        """Module with no fields shows ClassName()."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        assert repr(Empty()) == "Empty()"

    def test_groups_and_summary(self):
        """repr groups fields under headings and annotates each module with its size."""
        r = repr(nn.MLP([4, 16, 3], key=jax.random.key(0)))

        assert "MLP(  # 131 params, 524 B" in r
        assert "activation=relu, final_activation=None," in r
        assert "# Modules:" in r
        assert "# Parameters:" in r

    def test_frozen_summary(self):
        """Frozen params are counted in the summary and marked on the parameter."""
        r = repr(nn.Linear(4, 16, key=jax.random.key(0)).freeze())

        assert "80 frozen" in r
        assert "w=Param(float32(4, 16), frozen)," in r

    def test_plain_off_terminal(self):
        """Output captured to a pipe or file carries no escape sequences."""
        assert "\x1b" not in repr(nn.Linear(4, 16, key=jax.random.key(0)))

    def test_color_on_terminal(self, monkeypatch):
        """A color terminal highlights each class name with its Treescope hue."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        assert "\x1b[48;2;" in repr(nn.Linear(4, 16, key=jax.random.key(0)))

    def test_nesting_indents(self, monkeypatch):
        """Fields indent one level per module, and the closing bracket returns to its parent."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        plain = [
            re.sub(r"\x1b\[[0-9;]*m", "", line)
            for line in repr(nn.MLP([4, 16, 3], key=jax.random.key(0))).split("\n")
        ]
        assert plain[1] == "  activation=relu, final_activation=None,"
        assert plain[5] == "    w=Param(float32(4, 16)),"
        assert plain[7] == "  ),"
        assert plain[-1] == ")"

    def test_no_color_env_var(self, monkeypatch):
        """NO_COLOR suppresses highlighting even on a terminal."""
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        assert "\x1b" not in repr(nn.Linear(4, 16, key=jax.random.key(0)))


class TestFreezeUnfreeze:
    def test_freeze_sets_all_params_to_non_trainable(self):
        """Freezing a module sets all params to non-trainable."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3))

        m = Model().freeze()
        assert m.w.trainable is False
        assert m.b.trainable is False

    def test_unfreeze_sets_all_params_to_trainable(self):
        """Unfreezing a module sets all params to trainable."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3), trainable=False)

        m = Model().unfreeze()
        assert m.w.trainable is True

    def test_freeze_unfreeze_roundtrip(self):
        """Freeze then unfreeze preserves values and trainability."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))

        m = Model()
        npt.assert_array_equal(m.freeze().unfreeze().w._value, m.w._value)
        assert m.freeze().unfreeze().w.trainable is True

    def test_freeze_preserves_values(self):
        """Freezing preserves parameter values."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        m = Model(key=jax.random.key(0))
        frozen = m.freeze()
        npt.assert_array_equal(frozen.w._value, m.w._value)

    def test_freeze_nested_module(self):
        """Freezing a module freezes all nested child params."""

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

        m = Outer().freeze()
        assert m.inner.w.trainable is False
        assert m.b.trainable is False

    def test_partial_freeze_via_at(self):
        """Freeze one sub-module, keep the other trainable."""

        class Model(nn.Module):
            encoder: nn.Linear
            decoder: nn.Linear

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.encoder = nn.Linear(4, 8, key=keys[0])
                self.decoder = nn.Linear(8, 4, key=keys[1])

        m = Model(key=jax.random.key(0))
        m = m.at.encoder.set(m.encoder.freeze())
        assert m.encoder.w.trainable is False
        assert m.decoder.w.trainable is True

    def test_freeze_idempotent(self):
        """Freezing an already-frozen module is a no-op."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2), trainable=False)

        m = Model().freeze()
        assert m.w.trainable is False


class TestAstype:
    def test_astype_method(self):
        """Module.astype delegates to tree.astype correctly."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(2, dtype=jnp.float32))
                self.buf = jnp.zeros(2, dtype=jnp.float32)

        m = Model()
        via_method = m.astype(jnp.bfloat16)
        via_func = tree.astype(m, jnp.bfloat16)
        assert via_method.w._value.dtype == jnp.bfloat16
        assert via_method.buf.dtype == jnp.bfloat16
        npt.assert_allclose(via_method.w._value, via_func.w._value)
        npt.assert_allclose(via_method.buf, via_func.buf)


class TestNoneField:
    def test_module_with_none_field(self):
        """Module can store None as a field value (e.g., optional bias)."""
        linear = nn.Linear(4, 8, use_bias=False, key=jax.random.key(0))
        assert linear.b is None

    def test_none_field_survives_pytree_roundtrip(self):
        """None field is preserved through flatten/unflatten."""
        linear = nn.Linear(4, 8, use_bias=False, key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(linear)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.b is None
        npt.assert_array_equal(reconstructed.w._value, linear.w._value)

    def test_none_field_works_under_jit(self):
        """Module with a None field works correctly under jit."""
        linear = nn.Linear(4, 8, use_bias=False, key=jax.random.key(0))
        x = jnp.ones((1, 4))
        eager = linear(x)
        jitted = jax.jit(linear)(x)
        npt.assert_allclose(jitted, eager)


class TestContainerFields:
    def test_tuple_of_modules(self):
        """Module with a tuple of sub-modules."""
        keys = jax.random.split(jax.random.key(0), 2)
        seq = nn.Sequential(nn.Linear(4, 8, key=keys[0]), nn.Linear(8, 2, key=keys[1]))
        x = jnp.ones((1, 4))
        result = jax.jit(seq)(x)
        assert result.shape == (1, 2)

    def test_tuple_of_mixed_callables(self):
        """Sequential with Modules and plain functions."""
        keys = jax.random.split(jax.random.key(0), 2)
        seq = nn.Sequential(
            nn.Linear(4, 8, key=keys[0]),
            jax.nn.relu,
            nn.Linear(8, 2, key=keys[1]),
        )
        x = jnp.ones((1, 4))
        eager = seq(x)
        jitted = jax.jit(seq)(x)
        npt.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-5)

    def test_list_field(self):
        """Module with a list of sub-modules."""

        class Container(nn.Module):
            layers: list

            def __init__(self, key):
                keys = jax.random.split(key, 2)
                self.layers = [nn.Linear(4, 4, key=keys[0]), nn.Linear(4, 4, key=keys[1])]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        m = Container(key=jax.random.key(0))
        x = jnp.ones((1, 4))
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager)


class TestClassDefaultFields:
    def test_unassigned_default_field_works(self):
        """Custom __init__ can rely on a class-level default without assigning it."""

        class MyNorm(nn.Module):
            w: nn.Param
            eps: float = 1e-5

            def __init__(self, dim, *, key):
                self.w = nn.Param(jax.random.normal(key, (dim,)))

            def __call__(self, x):
                return self.w * x / (jnp.linalg.norm(x) + self.eps)

        m = MyNorm(4, key=jax.random.key(0))
        assert m.eps == 1e-5
        x = jnp.ones(4)
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-5)

    def test_default_field_survives_pytree_roundtrip(self):
        """Class-default fields are preserved through flatten/unflatten."""

        class WithDefault(nn.Module):
            w: nn.Param
            eps: float = 1e-5

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        m = WithDefault(key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(m)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert rebuilt.eps == 1e-5

    def test_missing_field_raises_clear_error(self):
        """A field with no default that __init__ never assigns raises at construction."""

        class NoDefault(nn.Module):
            w: nn.Param
            eps: float

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        with pytest.raises(AttributeError, match="'eps'.*never assigned"):
            NoDefault(key=jax.random.key(0))


class EncoderDecoder(NamedTuple):
    encoder: nn.Linear
    decoder: nn.Linear


class WithNamedTuple(nn.Module):
    pair: EncoderDecoder

    def __init__(self, key):
        keys = jax.random.split(key, 2)
        self.pair = EncoderDecoder(nn.Linear(4, 8, key=keys[0]), nn.Linear(8, 4, key=keys[1]))

    def __call__(self, x):
        return self.pair.decoder(self.pair.encoder(x))


class TestNamedTupleFields:
    def test_jit_matches_eager(self):
        """Module with a NamedTuple of sub-modules works under jit."""
        m = WithNamedTuple(key=jax.random.key(0))
        x = jnp.ones((1, 4))
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-5)

    def test_pytree_roundtrip(self):
        """Flatten/unflatten preserves the NamedTuple type and param values."""
        m = WithNamedTuple(key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(m)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert type(rebuilt.pair) is EncoderDecoder
        npt.assert_array_equal(rebuilt.pair.encoder.w._value, m.pair.encoder.w._value)
        npt.assert_array_equal(rebuilt.pair.decoder.w._value, m.pair.decoder.w._value)

    def test_grad_flows(self):
        """Gradients flow to Params inside a NamedTuple field."""
        m = WithNamedTuple(key=jax.random.key(0))
        x = jnp.ones((1, 4))
        grads = jax.grad(lambda m: jnp.sum(m(x) ** 2))(m)
        assert jnp.any(grads.pair.encoder.w._value != 0)
        assert jnp.any(grads.pair.decoder.w._value != 0)

    def test_freeze_through_namedtuple(self):
        """freeze() reaches Params inside a NamedTuple field."""
        m = WithNamedTuple(key=jax.random.key(0)).freeze()
        assert not m.pair.encoder.w.trainable
        assert not m.pair.decoder.w.trainable

    def test_mixed_static_element(self):
        """NamedTuple mixing a Module and a plain callable works under jit."""

        class Block(NamedTuple):
            linear: nn.Linear
            act: Callable

        class Container(nn.Module):
            block: Block

            def __init__(self, key):
                self.block = Block(nn.Linear(4, 4, key=key), jax.nn.relu)

            def __call__(self, x):
                return self.block.act(self.block.linear(x))

        m = Container(key=jax.random.key(0))
        x = jnp.ones((1, 4))
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-5)


class TestAtEdgeCases:
    def test_unknown_field_raises(self):
        """set raises AttributeError for unknown field names."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        with pytest.raises(AttributeError, match="has no field"):
            m.at.nonexistent.set(42)

    def test_navigating_into_leaf_raises(self):
        """Navigating into a non-container leaf raises TypeError."""
        linear = nn.Linear(4, 8, key=jax.random.key(0))
        with pytest.raises(TypeError, match="Cannot set"):
            linear.at.w.foo.set(1)

    def test_set_param(self):
        """set can swap a Param value."""
        linear = nn.Linear(4, 8, key=jax.random.key(0))
        new_w = nn.Param(jnp.zeros_like(linear.w._value))
        replaced = linear.at.w.set(new_w)
        npt.assert_array_equal(replaced.w._value, jnp.zeros_like(linear.w._value))
        # Original unchanged
        assert not jnp.allclose(linear.w._value, replaced.w._value)

    def test_set_param_to_non_param_breaks_tree_ops(self):
        """Setting a Param field to a plain value creates a structurally different pytree."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))

        m = Model()
        original_def = jax.tree.flatten(m)[1]

        # Replace Param with plain array
        m2 = m.at.w.set(jnp.ones(3))
        new_def = jax.tree.flatten(m2)[1]

        # The tree structures are different!
        assert original_def != new_def

    def test_changing_structure_works_under_jit(self):
        """internals.md: 'at can change pytree structure.' JIT will recompile this."""
        linear = nn.Linear(4, 8, key=jax.random.key(0))
        no_bias = linear.at.b.set(None)

        @jax.jit
        def f(m, x):
            return m(x)

        x = jnp.ones((1, 4))
        r1 = f(linear, x)
        r2 = f(no_bias, x)
        assert r1.shape == (1, 8)
        assert r2.shape == (1, 8)

    def test_set_none_changes_structure(self):
        """Setting a Param to None changes pytree structure (loses a leaf)."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3))

        m = Model()
        m2 = m.at.b.set(None)
        # Original has 2 leaves (w._value, b._value), replacement has 1
        assert len(jax.tree.leaves(m)) == 2
        assert len(jax.tree.leaves(m2)) == 1


class TestInheritance:
    def test_module_subclass_chain(self):
        """Subclass of a subclass works correctly."""

        class Base(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        class Child(Base):
            y: int

            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        c = Child(x=1, y=2)
        assert c.x == 1
        assert c.y == 2
        # Immutable
        with pytest.raises(AttributeError, match="frozen"):
            c.x = 3
        # Pytree roundtrip
        leaves, treedef = jax.tree.flatten(c)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.x == 1
        assert reconstructed.y == 2

    def test_super_init_does_not_freeze_early(self):
        """Child calling super().__init__() can still set its own fields."""

        class Base(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class Child(Base):
            b: nn.Param

            def __init__(self, key):
                super().__init__(key)
                self.b = nn.Param(jnp.zeros(4))

        c = Child(jax.random.key(0))
        assert c.w.shape == (4,)
        assert c.b.shape == (4,)
        # Frozen after construction
        with pytest.raises(AttributeError, match="frozen"):
            c.b = nn.Param(jnp.ones(4))

    def test_three_level_inheritance_with_super(self):
        """Three levels of inheritance using super().__init__()."""

        class Base(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class Mid(Base):
            b: nn.Param

            def __init__(self, key):
                super().__init__(key)
                self.b = nn.Param(jnp.zeros(4))

        class Top(Mid):
            scale: float

            def __init__(self, key):
                super().__init__(key)
                self.scale = 2.0

            def __call__(self, x):
                return (x @ self.w + self.b) * self.scale

        m = Top(jax.random.key(0))
        assert m.w.shape == (4,)
        assert m.b.shape == (4,)
        assert m.scale == 2.0
        # Works in jit
        result = jax.jit(m)(jnp.ones(4))
        assert result.shape == (4,)
        # Frozen after construction
        with pytest.raises(AttributeError, match="frozen"):
            m.scale = 3.0

    def test_inherited_module_pytree_roundtrip(self):
        """Inherited module survives flatten/unflatten correctly."""

        class Base(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class Child(Base):
            b: nn.Param

            def __init__(self, key):
                super().__init__(key)
                self.b = nn.Param(jnp.zeros(4))

        m = Child(jax.random.key(0))
        leaves, treedef = jax.tree.flatten(m)
        m2 = treedef.unflatten(leaves)
        npt.assert_array_equal(m2.w._value, m.w._value)
        npt.assert_array_equal(m2.b._value, m.b._value)

    def test_inherited_module_grad(self):
        """jax.grad works on an inherited module."""

        class Base(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

        class Child(Base):
            b: nn.Param

            def __init__(self, key):
                super().__init__(key)
                self.b = nn.Param(jnp.zeros(4))

        m = Child(jax.random.key(0))
        grads = jax.grad(lambda m: (m.w + m.b).sum())(m)
        npt.assert_allclose(grads.w._value, jnp.ones(4))
        npt.assert_allclose(grads.b._value, jnp.ones(4))


class TestParamsWithFrozen:
    def test_params_includes_frozen_params(self):
        """params property returns both trainable and frozen Params."""

        class Model(nn.Module):
            w: nn.Param
            b: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.b = nn.Param(jnp.zeros(3), trainable=False)

        m = Model()
        params = m.params
        assert isinstance(params.w, nn.Param)
        assert isinstance(params.b, nn.Param)
        assert params.w.trainable is True
        assert params.b.trainable is False

    def test_params_on_plain_array_field(self):
        """Plain array (non-Param) fields become None in params."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.buf = jnp.array([1.0, 2.0, 3.0])

        m = Model()
        params = m.params
        assert isinstance(params.w, nn.Param)
        assert params.buf is None

    def test_params_on_buffer_field(self):
        """Buffer fields become None in params rather than carrying mutable state."""
        params = nn.BatchNorm(3).params

        assert isinstance(params.scale, nn.Param)
        assert params.running_mean is None
        assert params.running_var is None


class TestDeepNesting:
    def test_three_level_nesting(self):
        """Three levels of module nesting work correctly."""

        class Inner(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))

        class Middle(nn.Module):
            inner: Inner

            def __init__(self, key):
                self.inner = Inner(key)

        class Outer(nn.Module):
            middle: Middle

            def __init__(self, key):
                self.middle = Middle(key)

        m = Outer(key=jax.random.key(0))
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 1
        # Roundtrip
        reconstructed = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        npt.assert_array_equal(reconstructed.middle.inner.w._value, m.middle.inner.w._value)
        # jit
        result = jax.jit(lambda m: jnp.sum(m.middle.inner.w))(m)
        npt.assert_allclose(result, jnp.sum(m.middle.inner.w._value))


class TestStaticWrapping:
    def test_jax_jit_with_callable_field(self):
        """Module with a callable field works under jax.jit."""

        class Model(nn.Module):
            w: nn.Param
            activation: Callable

            def __init__(self, *, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (2, 2)))
                self.activation = jax.nn.relu

        def forward(model, x):
            return model.activation(x @ model.w)

        model = Model(key=jax.random.key(0))
        x = jnp.ones((1, 2))
        eager = forward(model, x)
        jitted = jax.jit(forward)(model, x)
        npt.assert_allclose(jitted, eager)

    def test_jax_jit_with_int_field(self):
        """Module with an int field works under jax.jit."""

        class Model(nn.Module):
            w: nn.Param
            scale: int

            def __init__(self, *, key: jax.Array):
                self.w = nn.Param(jax.random.normal(key, (3,)))
                self.scale = 5

        def forward(model):
            return jnp.sum(model.w) * model.scale

        model = Model(key=jax.random.key(0))
        eager = forward(model)
        jitted = jax.jit(forward)(model)
        npt.assert_allclose(jitted, eager)

    def test_jax_grad_with_module(self):
        """jax.grad works directly on a module, producing gradients for all array leaves."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0, 3.0]))

        def loss(model):
            return jnp.sum(model.w * jnp.array([4.0, 5.0, 6.0]))

        grads = jax.grad(loss)(Model())
        npt.assert_allclose(grads.w._value, jnp.array([4.0, 5.0, 6.0]))


class TestModuleMutableFields:
    def test_list_field_can_be_mutated_in_place(self):
        """Module freezing doesn't prevent in-place mutation of mutable containers."""

        class Model(nn.Module):
            layers: list

            def __init__(self, key):
                self.layers = [nn.Linear(4, 4, key=key)]

        model = Model(key=jax.random.key(0))

        with pytest.raises(AttributeError, match="frozen"):
            model.layers = []

        # But in-place mutation bypasses the freeze
        original_len = len(model.layers)
        model.layers.append(nn.Linear(4, 4, key=jax.random.key(1)))
        assert len(model.layers) == original_len + 1

    def test_numpy_array_field_can_be_mutated(self):
        """numpy arrays in Module fields can be mutated in-place."""

        class Model(nn.Module):
            mask: np.ndarray
            w: nn.Param

            def __init__(self, key):
                self.mask = np.array([1.0, 0.0, 1.0])
                self.w = nn.Param(jax.random.normal(key, (3,)))

        model = Model(key=jax.random.key(0))

        original_val = model.mask[0]
        model.mask[0] = 999.0
        assert model.mask[0] == 999.0
        assert model.mask[0] != original_val


class TestModuleInheritanceEdgeCases:
    def test_module_with_no_fields(self):
        """Module subclass with no fields at all."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        m = Empty()
        assert jax.tree.leaves(m) == []
        reconstructed = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert isinstance(reconstructed, Empty)

    def test_abstract_base_then_concrete(self):
        """Abstract base Module with no fields, concrete child with fields."""

        class BaseLayer(nn.Module):
            def forward(self, x):
                raise NotImplementedError

        class ConcreteLayer(BaseLayer):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (4,)))

            def forward(self, x):
                return x @ self.w

        m = ConcreteLayer(key=jax.random.key(0))
        assert m.w.shape == (4,)
        leaves, treedef = jax.tree.flatten(m)
        m2 = treedef.unflatten(leaves)
        npt.assert_array_equal(m2.w._value, m.w._value)

    def test_sibling_classes_dont_interfere(self):
        """Two sibling Module subclasses with same field names don't interfere."""

        class A(nn.Module):
            x: int

            def __init__(self, x):
                self.x = x

        class B(nn.Module):
            x: int

            def __init__(self, x):
                self.x = x

        a = A(x=1)
        b = B(x=2)
        assert a.x == 1
        assert b.x == 2
        assert type(a) is not type(b)


class TestModuleCopy:
    def test_deepcopy_module_preserves_param_wrappers(self):
        """copy.deepcopy on Module preserves Param wrappers."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))

        m = Model()
        m2 = copy.deepcopy(m)
        assert isinstance(m2, Model)
        assert isinstance(m2.w, nn.Param)
        npt.assert_array_equal(m2.w._value, m.w._value)
        assert m2.w.trainable == m.w.trainable

    def test_copy_module_works(self):
        """copy.copy on Module works and preserves Param wrappers."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))

        m = Model()
        m2 = copy.copy(m)
        assert isinstance(m2.w, nn.Param)
        npt.assert_array_equal(m2.w._value, m.w._value)


class TestModuleWrappingEdgeCases:
    def test_numpy_scalar_treated_as_dynamic_leaf(self):
        """np.ndarray (even 0-d) is a dynamic leaf, not static metadata."""

        class Model(nn.Module):
            w: nn.Param
            buf: np.ndarray

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.buf = np.array(42.0)

        m = Model()
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 2

    def test_python_int_is_static_metadata(self):
        """Plain Python int becomes static aux (no dynamic leaf)."""

        class Model(nn.Module):
            w: nn.Param
            count: int

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))
                self.count = 42

        m = Model()
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 1

    def test_custom_class_field_becomes_static(self):
        """A custom (non-Module) class instance becomes static aux."""

        class Config:
            def __init__(self, lr):
                self.lr = lr

        class Model(nn.Module):
            w: nn.Param
            config: Config

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.config = Config(lr=0.001)

        m = Model(key=jax.random.key(0))
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 1
        reconstructed = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert reconstructed.config.lr == 0.001

    def test_dict_field_with_arrays_are_dynamic(self):
        """Arrays inside a dict field are dynamic leaves, not static."""

        class Model(nn.Module):
            w: nn.Param
            buffers: dict

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.buffers = {"mask": jnp.array([1.0, 0.0]), "scale": jnp.array(2.0)}

        m = Model(key=jax.random.key(0))
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 3

    def test_dict_field_mixed_static_dynamic(self):
        """Dict with mixed array and non-array values: arrays are dynamic, others are static."""

        class Model(nn.Module):
            w: nn.Param
            config: dict

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.config = {"lr": 0.001, "buf": jnp.array(1.0)}

        m = Model(key=jax.random.key(0))
        leaves = jax.tree.leaves(m)
        assert len(leaves) == 2

        reconstructed = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert reconstructed.config["lr"] == 0.001


class TestStatistics:
    def test_describes_distribution(self):
        """Each parameter gets a histogram and its moments."""
        from ion._rendering import statistics

        model = nn.Linear(64, 64, key=jax.random.key(0))
        described = statistics(model)

        assert any(block in described[id(model.w)] for block in "\u2581\u2588")
        assert "\u03bc=" in described[id(model.w)] and "\u03c3=" in described[id(model.w)]

    def test_constant_parameter(self):
        """A parameter with no width places its mass in the middle bucket, not the first."""
        from ion import _rendering
        from ion._rendering import statistics

        model = nn.Linear(64, 64, key=jax.random.key(0))

        edge = "\u2581" * (_rendering.BINS // 2)
        spike = f"{edge}\u2588{edge}"
        assert statistics(model)[id(model.b)] == f"{spike}  \u03bc=0 \u03c3=0"

    def test_low_precision_parameter(self):
        """Reductions run in float32, which bfloat16 and float8 scalars cannot format."""
        from ion._rendering import statistics

        for dtype in (jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn):
            model = nn.Linear(8, 64, key=jax.random.key(0)).astype(dtype)

            assert "\u03bc=" in statistics(model)[id(model.w)]

    def test_annotations_share_a_column(self):
        """Descriptions align down a group so distributions can be compared by eye."""
        from ion._rendering import module_repr, statistics

        model = nn.Linear(8, 64, key=jax.random.key(0))
        lines = module_repr(model, statistics(model)).split("\n")[2:4]

        columns = {line.index("\u03bc=") for line in lines}
        assert len(columns) == 1

    def test_repr_stays_free_of_statistics(self):
        """Plain repr does no reductions, keeping logging and debuggers cheap."""
        r = repr(nn.Linear(64, 64, key=jax.random.key(0)))

        assert "\u03bc=" not in r
        assert not any(block in r for block in "\u2588\u2587\u2586")


class TestReprInsideTransformations:
    """Test that __repr__ doesn't crash inside JAX transformations."""

    def test_repr_inside_jit(self):
        """Calling repr() on a module inside jit doesn't crash."""

        class Model(nn.Module):
            w: nn.Param
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.scale = 2.0

        m = Model()

        @jax.jit
        def f(m):
            _ = repr(m)
            return jnp.sum(m.w)

        result = f(m)
        npt.assert_allclose(result, 3.0)

    def test_repr_inside_vmap(self):
        """Calling repr() on a module inside vmap doesn't crash."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))

        m = Model()

        def f(m):
            _ = repr(m)
            return jnp.sum(m.w)

        # vmap over a batch dim added to the param
        batched_m = jax.tree.map(lambda x: jnp.stack([x, x]), m)
        results = jax.vmap(f)(batched_m)
        npt.assert_allclose(results, jnp.array([3.0, 3.0]))


class TestStaticFieldTypes:
    """Test that various field types (list, dict, tuple, callable) work through jit."""

    def test_list_of_ints_through_jit(self):
        """List of ints field works as static metadata through jit."""

        class Model(nn.Module):
            w: nn.Param
            dims: list

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.dims = [1, 2, 3]

            def __call__(self, x):
                return x * self.dims[0]

        m = Model()
        x = jnp.array(5.0)
        npt.assert_allclose(jax.jit(m)(x), 5.0)

    def test_dict_of_scalars_through_jit(self):
        """Dict of scalars field works as static metadata through jit."""

        class Model(nn.Module):
            w: nn.Param
            config: dict

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.config = {"scale": 2.0, "offset": 1.0}

            def __call__(self, x):
                return x * self.config["scale"] + self.config["offset"]

        m = Model()
        x = jnp.array(3.0)
        npt.assert_allclose(jax.jit(m)(x), 7.0)

    def test_tuple_of_callables_through_jit(self):
        """Tuple of callables field works as static metadata through jit."""

        class Model(nn.Module):
            w: nn.Param
            activations: tuple

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.activations = (jax.nn.relu, jax.nn.sigmoid)

            def __call__(self, x):
                for act in self.activations:
                    x = act(x)
                return x

        m = Model()
        x = jnp.array([-1.0, 0.0, 1.0])
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager)

    def test_nested_dict_with_arrays_through_jit(self):
        """Dict mixing arrays and scalars: arrays are dynamic, scalars are static."""

        class Model(nn.Module):
            w: nn.Param
            meta: dict

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.meta = {"scale": 2.0, "mask": jnp.array([1.0, 0.0, 1.0])}

            def __call__(self, x):
                return x * self.meta["mask"] * self.meta["scale"]

        m = Model()
        x = jnp.array([1.0, 2.0, 3.0])
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager)

    def test_mutating_static_list_uses_stale_cache(self):
        """Sharp edge: in-place mutation of a static list hits the stale trace, no retrace."""

        class Model(nn.Module):
            w: nn.Param
            dims: list

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.dims = [1]

            def __call__(self, x):
                return x * len(self.dims)

        m = Model()
        x = jnp.array(1.0)

        result1 = jax.jit(m)(x)
        npt.assert_allclose(result1, 1.0)

        # Mutate the list in-place, JAX does NOT retrace
        m.dims.append(2)
        result2 = jax.jit(m)(x)
        # Stale! Still returns 1.0, not 2.0, because the old trace is cached
        npt.assert_allclose(result2, 1.0)


class TestPytreeSharedReferences:
    """internals.md: 'JAX pytrees are trees, not graphs'."""

    def test_shared_module_is_duplicated_through_flatten_unflatten(self):
        """Shared sub-module reference becomes two independent copies after roundtrip."""

        class Inner(nn.Module):
            w: nn.Param

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))

        class Shared(nn.Module):
            a: Inner
            b: Inner

            def __init__(self, layer):
                self.a = layer
                self.b = layer

        layer = Inner(jax.random.key(0))
        model = Shared(layer)
        assert model.a is model.b

        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(model)))
        assert rebuilt.a is not rebuilt.b
        npt.assert_array_equal(rebuilt.a.w._value, rebuilt.b.w._value)

    def test_shared_param_is_duplicated(self):
        """Shared Param reference becomes two independent copies after roundtrip."""

        class SharedParam(nn.Module):
            a: nn.Param
            b: nn.Param

            def __init__(self, p):
                self.a = p
                self.b = p

        p = nn.Param(jnp.ones(3))
        model = SharedParam(p)
        assert model.a is model.b

        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(model)))
        assert rebuilt.a is not rebuilt.b

    def test_weight_tying_via_array_reference(self):
        """Weight tying via direct array reference works and grads flow through."""

        class TiedModel(nn.Module):
            embed: nn.Param
            hidden: nn.Param

            def __init__(self, key: jax.Array):
                self.embed = nn.Param(jax.random.normal(key, (4, 8)))
                self.hidden = nn.Param(jnp.zeros(8))

            def decode(self, h):
                return h @ self.embed.T

        model = TiedModel(key=jax.random.key(0))
        h = jnp.ones(8)
        out = model.decode(h)
        assert out.shape == (4,)

        grads = jax.grad(lambda m, h: jnp.sum(m.decode(h)))(model, h)
        assert jnp.any(grads.embed._value != 0)


class TestParamsPropertyEdgeCases:
    def test_params_replaces_plain_array_with_none_but_keeps_static(self):
        """Plain arrays become None but static fields are preserved (by design)."""

        class Model(nn.Module):
            w: nn.Param
            buf: jax.Array
            scale: float

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.buf = jnp.ones(3)
                self.scale = 2.0

        m = Model()
        p = m.params
        assert isinstance(p.w, nn.Param)
        assert p.buf is None
        assert p.scale == 2.0

    def test_params_with_callable_field(self):
        """internals.md: 'Module.params preserves static fields alongside Param leaves'."""

        class Model(nn.Module):
            w: nn.Param
            act: Callable

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.act = jax.nn.relu

        m = Model()
        p = m.params
        assert isinstance(p.w, nn.Param)
        assert p.act is jax.nn.relu

    def test_params_with_nested_module_plain_array(self):
        """Nested module's plain array fields also become None."""

        class Inner(nn.Module):
            w: nn.Param
            buf: jax.Array

            def __init__(self, key):
                self.w = nn.Param(jax.random.normal(key, (2,)))
                self.buf = jnp.ones(2)

        class Outer(nn.Module):
            inner: Inner

            def __init__(self, key):
                self.inner = Inner(key)

        m = Outer(key=jax.random.key(0))
        p = m.params
        assert isinstance(p.inner.w, nn.Param)
        assert p.inner.buf is None


class TestFieldPartitioning:
    """Tests for the dynamic child vs static aux partitioning in _register_module."""

    def test_set_param_to_none_changes_treedef(self):
        """Setting a Param field to None moves it from dynamic child to static aux."""
        model = nn.Linear(3, 4, key=jax.random.key(0))
        _, treedef_with_bias = jax.tree.flatten(model)

        model_no_bias = model.at.b.set(None)
        leaves, treedef_no_bias = jax.tree.flatten(model_no_bias)

        # Different treedef (b moved from child to static aux)
        assert treedef_with_bias != treedef_no_bias
        # Only w remains as a leaf
        assert len(leaves) == 1
        # Roundtrip preserves None
        rebuilt = treedef_no_bias.unflatten(leaves)
        assert rebuilt.b is None
        npt.assert_array_equal(rebuilt.w._value, model_no_bias.w._value)

    def test_set_none_to_param_changes_treedef(self):
        """Setting a None field to a Param moves it from static aux to dynamic child."""
        model = nn.Linear(3, 4, use_bias=False, key=jax.random.key(0))
        assert model.b is None
        leaves_before = jax.tree.leaves(model)
        assert len(leaves_before) == 1

        new_bias = nn.Param(jnp.zeros(4))
        model_with_bias = model.at.b.set(new_bias)
        leaves_after = jax.tree.leaves(model_with_bias)
        assert len(leaves_after) == 2

    def test_surgery_both_versions_work_through_jit(self):
        """Both pre- and post-surgery models work through jit."""
        model = nn.Linear(3, 4, key=jax.random.key(0))
        x = jnp.ones((2, 3))

        result_with_bias = jax.jit(model)(x)
        assert result_with_bias.shape == (2, 4)

        model_no_bias = model.at.b.set(None)
        result_no_bias = jax.jit(model_no_bias)(x)
        assert result_no_bias.shape == (2, 4)

    def test_mixed_tuple_modules_and_callable_through_jit_grad(self):
        """Tuple mixing Modules and callables: Modules are dynamic, callables wrapped in _Static."""

        class Model(nn.Module):
            layers: tuple

            def __init__(self, key):
                self.layers = (nn.Linear(3, 3, key=key), jax.nn.relu)

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        m = Model(key=jax.random.key(0))
        x = jnp.ones((2, 3))

        # Works through jit
        eager = m(x)
        jitted = jax.jit(m)(x)
        npt.assert_allclose(jitted, eager)

        # Works through grad
        grads = jax.grad(lambda m, x: jnp.sum(m(x)))(m, x)
        assert isinstance(grads.layers[0], nn.Linear)
        # Callable preserved through roundtrip
        assert grads.layers[1] is jax.nn.relu

    def test_empty_tuple_field_is_static(self):
        """Empty tuple has no array-like elements, so it goes to static aux."""

        class Model(nn.Module):
            w: nn.Param
            items: tuple

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.items = ()

        m = Model()
        assert len(jax.tree.leaves(m)) == 1
        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert rebuilt.items == ()

    def test_empty_list_field_is_static(self):
        """Empty list has no array-like elements, so it goes to static aux."""

        class Model(nn.Module):
            w: nn.Param
            items: list

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.items = []

        m = Model()
        assert len(jax.tree.leaves(m)) == 1
        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert rebuilt.items == []

    def test_empty_dict_field_is_static(self):
        """Empty dict has no array-like values, so it goes to static aux."""

        class Model(nn.Module):
            w: nn.Param
            meta: dict

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.meta = {}

        m = Model()
        assert len(jax.tree.leaves(m)) == 1
        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert rebuilt.meta == {}

    def test_none_field_is_static(self):
        """None field goes to static aux, not dynamic children."""

        class Model(nn.Module):
            w: nn.Param
            optional: None

            def __init__(self):
                self.w = nn.Param(jnp.ones(3))
                self.optional = None

        m = Model()
        assert len(jax.tree.leaves(m)) == 1
        rebuilt = jax.tree.unflatten(*reversed(jax.tree.flatten(m)))
        assert rebuilt.optional is None
        npt.assert_allclose(jax.jit(lambda m: jnp.sum(m.w))(m), 3.0)
