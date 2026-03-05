import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

from ion import nn


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
                k1, k2 = jax.random.split(key)
                self.w = nn.Param(jax.random.normal(k1, (3, 4)))
                self.b = nn.Param(jnp.zeros(4))

        m = Model(key=jax.random.key(0))
        leaves, treedef = jtu.tree_flatten(m)
        reconstructed = treedef.unflatten(leaves)
        npt.assert_array_equal(reconstructed.w.value, m.w.value)
        npt.assert_array_equal(reconstructed.b.value, m.b.value)

    def test_children_follow_field_order(self):
        """Non-array fields are wrapped as Static, so leaves are empty, but roundtrip preserves values."""

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
        reconstructed = jtu.tree_unflatten(*reversed(jtu.tree_flatten(m)))
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
        leaves, treedef = jtu.tree_flatten(m)
        # This would fail if unflatten tried to call __init__(in_dim, out_dim, key)
        reconstructed = treedef.unflatten(leaves)
        npt.assert_array_equal(reconstructed.w.value, m.w.value)

    def test_tree_map(self):
        """jax.tree.map transforms leaves inside a module."""

        class Model(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.array([1.0, 2.0]))

        m = Model()
        doubled = jax.tree.map(lambda x: x * 2, m)
        assert isinstance(doubled, Model)
        npt.assert_array_equal(doubled.w.value, jnp.array([2.0, 4.0]))

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


class TestReplace:
    def test_single_field(self):
        """Replacing one field returns correct values."""

        class Model(nn.Module):
            a: int
            b: int

            def __init__(self, a: int, b: int):
                self.a = a
                self.b = b

        m = Model(a=1, b=2)
        m2 = m.replace(b=10)
        assert m2.a == 1
        assert m2.b == 10

    def test_multiple_fields(self):
        """Replacing several fields at once."""

        class Model(nn.Module):
            a: int
            b: int
            c: int

            def __init__(self, a: int, b: int, c: int):
                self.a = a
                self.b = b
                self.c = c

        m = Model(a=1, b=2, c=3)
        m2 = m.replace(a=10, c=30)
        assert m2.a == 10
        assert m2.b == 2
        assert m2.c == 30

    def test_returns_new_instance(self):
        """Original module is unchanged after replace."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        m2 = m.replace(x=2)
        assert m.x == 1
        assert m2.x == 2
        assert m is not m2

    def test_replaced_is_frozen(self):
        """The copy returned by replace is also immutable."""

        class Model(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Model(x=1)
        m2 = m.replace(x=2)
        with pytest.raises(AttributeError, match="frozen"):
            m2.x = 3

    def test_preserves_type(self):
        """replace returns the same subclass type."""

        class Child(nn.Module):
            x: int

            def __init__(self, x: int):
                self.x = x

        m = Child(x=1)
        m2 = m.replace(x=2)
        assert type(m2) is Child


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
                k1, k2 = jax.random.split(key)
                self.layer1 = nn.Linear(4, 8, key=k1)
                self.layer2 = nn.Linear(8, 2, key=k2)

        m = Container(key=key)
        param_leaves = jax.tree.leaves(m.params)
        # Linear has w (Param) and b (Param), so 2 layers * 2 params = 4
        assert len(param_leaves) == 4
        for leaf in param_leaves:
            assert isinstance(leaf, jnp.ndarray)


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
        """repr contains abbreviated dtype and shape for plain array fields."""

        class Model(nn.Module):
            x: jax.Array

            def __init__(self):
                self.x = jnp.zeros((3, 4), dtype=jnp.float32)

        r = repr(Model())
        assert "f32" in r
        assert "[3, 4]" in r

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

        class Container(nn.Module):
            layers: tuple

            def __init__(self):
                self.layers = (nn.Identity(), nn.Identity())

        r = repr(Container())
        assert "(" in r  # tuple open bracket
        assert "Identity()" in r

    def test_empty_module(self):
        """Module with no fields shows ClassName()."""

        class Empty(nn.Module):
            def __init__(self):
                pass

        assert repr(Empty()) == "Empty()"


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
        npt.assert_allclose(grads.w.value, jnp.array([4.0, 5.0, 6.0]))
