from ion import gnn
from ion import nn
import jax
import jax.numpy as jnp
import math
import numpy.testing as npt
import pytest


class TestRGCNConv:
    def test_output_manual(self):
        """Output matches per-relation mean aggregation plus the root transform."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        edge_type = jnp.array([0, 1, 0])

        expected = x @ conv.w_root + conv.b  # type: ignore[operator]
        for relation in range(2):
            mask = edge_type == relation
            neigh = (x @ conv.w_neigh[relation])[senders] * mask[:, None]
            count = gnn.segment_sum(mask.astype(x.dtype), receivers, 3)
            expected = (
                expected
                + gnn.segment_sum(neigh, receivers, 3) / jnp.where(count > 0, count, 1)[:, None]
            )

        y = conv(x, senders, receivers, edge_type=edge_type)

        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_single_relation_matches_sage_mean(self):
        """One relation reduces to mean aggregation, matching SAGEConv."""
        sage = gnn.SAGEConv(4, 6, key=jax.random.key(0))
        conv = gnn.RGCNConv(4, 6, 1, key=jax.random.key(1))
        conv = conv.at.w_neigh.set(nn.Param(sage.w_neigh.value[None]))
        conv = conv.at.w_root.set(sage.w_root)
        conv = conv.at.b.set(sage.b)
        x = jax.random.normal(jax.random.key(2), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        y = conv(x, senders, receivers, edge_type=jnp.zeros(3, dtype=jnp.int32))

        npt.assert_allclose(y, sage(x, senders, receivers), rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        conv = gnn.RGCNConv(8, 16, 3, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        edge_type = jnp.array([0, 1, 2, 0])

        y = conv(x, senders, receivers, edge_type=edge_type)

        assert y.shape == (5, 16)

    def test_relations_use_separate_transforms(self):
        """Relabelling an edge's relation changes the output it contributes to."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])

        first = conv(x, senders, receivers, edge_type=jnp.array([0]))
        second = conv(x, senders, receivers, edge_type=jnp.array([1]))

        assert not jnp.allclose(first[1], second[1])

    def test_basis_decomposition_shapes(self):
        """num_bases shares matrices across relations through mixing coefficients."""
        conv = gnn.RGCNConv(4, 6, 12, num_bases=2, key=jax.random.key(0))

        assert conv.w_neigh.shape == (2, 4, 6)
        assert conv.w_coeff is not None
        assert conv.w_coeff.shape == (12, 2)

    def test_no_basis_decomposition(self):
        """Without num_bases each relation holds its own transform."""
        conv = gnn.RGCNConv(4, 6, 3, key=jax.random.key(0))

        assert conv.w_neigh.shape == (3, 4, 6)
        assert conv.w_coeff is None

    def test_no_bias(self):
        """use_bias=False removes the bias."""
        conv = gnn.RGCNConv(4, 6, 2, use_bias=False, key=jax.random.key(0))

        assert conv.b is None

    def test_isolated_node_gets_root_and_bias(self):
        """A node with no incoming edges receives only its root term and bias."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])

        y = conv(x, senders, receivers, edge_type=jnp.array([0]))
        expected = x[2] @ conv.w_root + conv.b  # type: ignore[operator]

        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-5)

    def test_jit_and_grad(self):
        """The layer traces under jit and produces finite parameter gradients."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        edge_type = jnp.array([0, 1, 0])

        y = conv(x, senders, receivers, edge_type=edge_type)
        jitted = jax.jit(conv)(x, senders, receivers, edge_type=edge_type)
        grads = jax.grad(lambda m: m(x, senders, receivers, edge_type=edge_type).sum())(conv)

        npt.assert_allclose(jitted, y, rtol=1e-5, atol=1e-5)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))

    def test_default_dtype(self):
        """Weights default to float32."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))

        assert conv.w_neigh.dtype == jnp.float32
        assert conv.w_root.dtype == jnp.float32


def _graph():
    """Four nodes over two types, five edges over two relations."""
    x = jax.random.normal(jax.random.key(1), (4, 6))
    node_type = jnp.array([0, 1, 0, 1])
    senders = jnp.array([0, 1, 2, 3, 0])
    receivers = jnp.array([1, 2, 3, 0, 2])
    edge_type = jnp.array([0, 1, 0, 1, 1])
    return x, node_type, senders, receivers, edge_type


class TestHGTConv:
    def test_output_manual(self):
        """Output matches the type-dependent attention equations computed per edge."""
        conv = gnn.HGTConv(6, 6, 2, 2, num_heads=2, key=jax.random.key(0))
        x, node_type, senders, receivers, edge_type = _graph()
        heads, head_dim = 2, 3
        assert conv.b_out is not None

        k = jnp.stack([x[i] @ conv.w_k[node_type[i]] for i in range(4)]).reshape(4, heads, head_dim)
        q = jnp.stack([x[i] @ conv.w_q[node_type[i]] for i in range(4)]).reshape(4, heads, head_dim)
        v = jnp.stack([x[i] @ conv.w_v[node_type[i]] for i in range(4)]).reshape(4, heads, head_dim)

        logits = jnp.array(
            [
                [
                    (k[s, h] @ conv.w_att[r, h] * q[t, h]).sum() * conv.mu[r, h] / math.sqrt(3)
                    for h in range(heads)
                ]
                for s, t, r in zip(senders, receivers, edge_type)
            ]
        )
        attention = gnn.segment_softmax(logits, receivers, 4)
        messages = jnp.stack(
            [
                jnp.stack([v[s, h] @ conv.w_msg[r, h] for h in range(heads)])
                for s, r in zip(senders, edge_type)
            ]
        )
        agg = gnn.segment_sum(messages * attention[..., None], receivers, 4).reshape(4, -1)
        expected = jnp.stack(
            [
                jax.nn.gelu(agg[i]) @ conv.w_out[node_type[i]] + conv.b_out[node_type[i]]
                for i in range(4)
            ]
        )
        gate = jax.nn.sigmoid(conv.skip[node_type])[:, None]  # type: ignore[index]
        expected = gate * expected + (1 - gate) * x

        y = conv(x, senders, receivers, node_type=node_type, edge_type=edge_type)

        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_attention_normalizes_across_relations(self):
        """Every incoming edge of a node competes in one softmax, whatever its relation."""
        conv = gnn.HGTConv(6, 6, 2, 2, num_heads=2, key=jax.random.key(0))
        x, node_type, senders, receivers, edge_type = _graph()

        k = jnp.einsum("ni,tio->nto", x, conv.w_k)[jnp.arange(4), node_type].reshape(4, 2, 3)
        q = jnp.einsum("ni,tio->nto", x, conv.w_q)[jnp.arange(4), node_type].reshape(4, 2, 3)
        edge_k = jnp.einsum("nhd,rhdf->nrhf", k, conv.w_att)[senders, edge_type]
        logits = (q[receivers] * edge_k).sum(axis=-1) * conv.mu[edge_type] / math.sqrt(3)
        attention = gnn.segment_softmax(logits, receivers, 4)

        npt.assert_allclose(gnn.segment_sum(attention, receivers, 4), jnp.ones((4, 2)), atol=1e-5)

    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        conv = gnn.HGTConv(6, 8, 2, 2, num_heads=2, use_skip=False, key=jax.random.key(0))
        x, node_type, senders, receivers, edge_type = _graph()

        y = conv(x, senders, receivers, node_type=node_type, edge_type=edge_type)

        assert y.shape == (4, 8)

    def test_relations_use_separate_transforms(self):
        """Relabelling an edge's relation changes the output it contributes to."""
        conv = gnn.HGTConv(6, 6, 2, 2, num_heads=2, key=jax.random.key(0))
        x, node_type, _, _, _ = _graph()
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 1])

        first = conv(x, senders, receivers, node_type=node_type, edge_type=jnp.array([0, 0]))
        second = conv(x, senders, receivers, node_type=node_type, edge_type=jnp.array([0, 1]))

        assert not jnp.allclose(first[1], second[1])

    def test_node_types_use_separate_transforms(self):
        """Relabelling a node's type changes its own projection."""
        conv = gnn.HGTConv(6, 6, 2, 2, num_heads=2, key=jax.random.key(0))
        x, _, senders, receivers, edge_type = _graph()

        first = conv(x, senders, receivers, node_type=jnp.zeros(4, int), edge_type=edge_type)
        second = conv(x, senders, receivers, node_type=jnp.array([0, 1, 0, 0]), edge_type=edge_type)

        assert not jnp.allclose(first[1], second[1])

    def test_skip_requires_matching_dims(self):
        """The learned skip gate mixes with the input, so it needs in_dim == out_dim."""
        with pytest.raises(ValueError, match="use_skip=True requires"):
            gnn.HGTConv(6, 8, 2, 2, key=jax.random.key(0))

    def test_no_skip(self):
        """use_skip=False removes the gate and lets the layer change width."""
        conv = gnn.HGTConv(6, 8, 2, 2, use_skip=False, key=jax.random.key(0))

        assert conv.skip is None

    def test_no_bias(self):
        """use_bias=False removes the per-type output bias."""
        conv = gnn.HGTConv(6, 6, 2, 2, use_bias=False, key=jax.random.key(0))

        assert conv.b_out is None

    def test_prior_and_gate_init_to_ones(self):
        """The relation prior and skip gate start at one, as in the paper's reference code."""
        conv = gnn.HGTConv(6, 6, 2, 3, num_heads=2, key=jax.random.key(0))

        assert jnp.all(conv.mu == 1)
        assert jnp.all(conv.skip == 1)  # type: ignore[arg-type]

    def test_jit_and_grad(self):
        """The layer traces under jit and produces finite parameter gradients."""
        conv = gnn.HGTConv(6, 6, 2, 2, num_heads=2, key=jax.random.key(0))
        x, node_type, senders, receivers, edge_type = _graph()

        y = conv(x, senders, receivers, node_type=node_type, edge_type=edge_type)
        jitted = jax.jit(conv)(x, senders, receivers, node_type=node_type, edge_type=edge_type)
        grads = jax.grad(
            lambda m: m(x, senders, receivers, node_type=node_type, edge_type=edge_type).sum()
        )(conv)

        npt.assert_allclose(jitted, y, rtol=1e-5, atol=1e-5)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))

    def test_default_dtype(self):
        """Weights default to float32."""
        conv = gnn.HGTConv(6, 6, 2, 2, key=jax.random.key(0))

        assert conv.w_k.dtype == jnp.float32
        assert conv.w_att.dtype == jnp.float32
