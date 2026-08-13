import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn, nn


class TestGraphNetwork:
    def test_output_manual(self):
        """Outputs match a manually written edge, aggregate, and node update."""
        key_edge, key_node, key_x, key_x_edge = jax.random.split(jax.random.key(0), 4)
        edge_model = nn.MLP([10, 8, 6], key=key_edge)
        node_model = nn.MLP([10, 8, 5], key=key_node)
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
        x = jax.random.normal(key_x, (3, 4))
        x_edge = jax.random.normal(key_x_edge, (3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        edge_inputs = jnp.concatenate((x[senders], x[receivers], x_edge), axis=-1)
        edge_expected = edge_model(edge_inputs)
        received = gnn.segment_sum(edge_expected, receivers, 3)
        node_expected = node_model(jnp.concatenate((x, received), axis=-1))

        node_result, edge_result = network(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_result, node_expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(edge_result, edge_expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite inputs update destination nodes from separate source nodes."""
        key_edge, key_node, key_src, key_dst, key_x_edge = jax.random.split(jax.random.key(0), 5)
        edge_model = nn.MLP([10, 4], key=key_edge)
        node_model = nn.MLP([9, 7], key=key_node)
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
        x_src = jax.random.normal(key_src, (4, 3))
        x_dst = jax.random.normal(key_dst, (2, 5))
        x_edge = jax.random.normal(key_x_edge, (3, 2))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        edge_inputs = jnp.concatenate((x_src[senders], x_dst[receivers], x_edge), axis=-1)
        edge_expected = edge_model(edge_inputs)
        received = gnn.segment_sum(edge_expected, receivers, 2)
        node_expected = node_model(jnp.concatenate((x_dst, received), axis=-1))

        node_result, edge_result = network((x_src, x_dst), senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_result, node_expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(edge_result, edge_expected, rtol=1e-5, atol=1e-5)

    def test_without_input_edge_features(self):
        """The edge model can create edge features from the incident nodes alone."""
        edge_model = nn.MLP([8, 6], key=jax.random.key(0))
        node_model = nn.MLP([10, 5], key=jax.random.key(1))
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
        x = jax.random.normal(jax.random.key(2), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        node_out, edge_out = network(x, senders, receivers)

        assert node_out.shape == (3, 5)
        assert edge_out.shape == (3, 6)

    def test_segment_mean(self):
        """A compatible non-default aggregation callable controls reduction."""
        edge_model = nn.MLP([10, 6], key=jax.random.key(0))
        node_model = nn.MLP([10, 5], key=jax.random.key(1))
        network = gnn.GraphNetwork(
            edge_model=edge_model,
            node_model=node_model,
            aggregate=gnn.segment_mean,
        )
        x = jax.random.normal(jax.random.key(2), (3, 4))
        x_edge = jax.random.normal(jax.random.key(3), (4, 2))
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([2, 2, 0, 1])

        edge_out = edge_model(jnp.concatenate((x[senders], x[receivers], x_edge), axis=-1))
        received = gnn.segment_mean(edge_out, receivers, 3)
        expected = node_model(jnp.concatenate((x, received), axis=-1))
        node_out, _ = network(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_out, expected, rtol=1e-5, atol=1e-5)

    def test_isolated_node_receives_zero_sum(self):
        """The default aggregation gives isolated nodes a zero message."""
        network = gnn.GraphNetwork(
            edge_model=lambda inputs: inputs[:, :2],
            node_model=lambda inputs: inputs[:, -2:],
        )
        x = jax.random.normal(jax.random.key(0), (3, 4))

        node_out, _ = network(x, jnp.array([0]), jnp.array([1]))

        npt.assert_array_equal(node_out[2], jnp.zeros(2))

    def test_updated_edges_are_aggregated(self):
        """Node updates receive new edge values rather than previous edge values."""
        network = gnn.GraphNetwork(
            edge_model=lambda inputs: inputs[:, -1:] + 1,
            node_model=lambda inputs: inputs[:, -1:],
        )
        x = jnp.zeros((2, 1))
        x_edge = jnp.array([[2.0], [3.0]])
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 1])

        node_out, edge_out = network(x, senders, receivers, x_edge=x_edge)

        npt.assert_array_equal(edge_out, jnp.array([[3.0], [4.0]]))
        npt.assert_array_equal(node_out[1], jnp.array([7.0]))

    def test_edge_features_change_edges_and_nodes(self):
        """Initial edge features affect returned edges and downstream nodes."""
        edge_model = nn.MLP([10, 6], key=jax.random.key(0))
        node_model = nn.MLP([10, 5], key=jax.random.key(1))
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
        x = jax.random.normal(jax.random.key(2), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        node_zero, edge_zero = network(x, senders, receivers, x_edge=jnp.zeros((3, 2)))
        node_edge, edge_edge = network(x, senders, receivers, x_edge=jnp.ones((3, 2)))

        assert not jnp.allclose(edge_zero, edge_edge)
        assert not jnp.allclose(node_zero, node_edge)

    def test_creates_no_params(self):
        """All parameters belong to the caller-supplied edge and node models."""
        edge_model = nn.MLP([10, 8, 6], key=jax.random.key(0))
        node_model = nn.MLP([10, 8, 5], key=jax.random.key(1))
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)

        assert network.num_params == edge_model.num_params + node_model.num_params

    def test_matches_edge_update_then_node_update(self):
        """The block is exactly EdgeUpdate composed with NodeUpdate."""
        key_edge, key_node, key_x, key_x_edge = jax.random.split(jax.random.key(0), 4)
        edge_model = nn.MLP([10, 8, 6], key=key_edge)
        node_model = nn.MLP([10, 8, 5], key=key_node)
        network = gnn.GraphNetwork(edge_model=edge_model, node_model=node_model)
        x = jax.random.normal(key_x, (3, 4))
        x_edge = jax.random.normal(key_x_edge, (3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        edge_expected = gnn.EdgeUpdate(edge_model)(x, senders, receivers, x_edge=x_edge)
        node_expected = gnn.NodeUpdate(node_model)(x, senders, receivers, x_edge=edge_expected)

        node_result, edge_result = network(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_result, node_expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(edge_result, edge_expected, rtol=1e-5, atol=1e-5)

    def test_models_are_keyword_only(self):
        """Positional models are rejected to prevent swapping the two callables."""
        edge_model = nn.MLP([10, 6], key=jax.random.key(0))
        node_model = nn.MLP([10, 5], key=jax.random.key(1))

        with pytest.raises(TypeError):
            gnn.GraphNetwork(edge_model, node_model)  # type: ignore[call-arg]


class TestEdgeUpdate:
    def test_output_manual(self):
        """Output is edge_model(concat(sender, receiver, edge))."""
        edge_model = nn.MLP([10, 8, 6], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        expected = edge_model(jnp.concatenate((x[senders], x[receivers], x_edge), axis=-1))
        result = update(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite edges gather endpoints from separate node sets."""
        edge_model = nn.MLP([10, 7], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        x_edge = jax.random.normal(jax.random.key(3), (3, 2))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        edge_inputs = jnp.concatenate((x_src[senders], x_dst[receivers], x_edge), axis=-1)
        expected = edge_model(edge_inputs)
        y = update((x_src, x_dst), senders, receivers, x_edge=x_edge)

        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        """Output has one row per edge and the edge model's output dimension."""
        update = gnn.EdgeUpdate(nn.MLP([10, 6], key=jax.random.key(0)))
        x = jnp.ones((5, 4))
        x_edge = jnp.ones((3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 3])

        assert update(x, senders, receivers, x_edge=x_edge).shape == (3, 6)

    def test_without_input_edge_features(self):
        """Omitting x_edge builds edge features from the incident nodes alone."""
        edge_model = nn.MLP([8, 6], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        expected = edge_model(jnp.concatenate((x[senders], x[receivers]), axis=-1))

        npt.assert_allclose(update(x, senders, receivers), expected, rtol=1e-5, atol=1e-5)

    def test_creates_no_params(self):
        """All parameters belong to the caller-supplied edge model."""
        edge_model = nn.MLP([10, 8, 6], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)

        assert update.num_params == edge_model.num_params


class TestNodeUpdate:
    def test_output_manual(self):
        """Output is node_model(concat(node, aggregate(edge features)))."""
        node_model = nn.MLP([10, 8, 5], key=jax.random.key(0))
        update = gnn.NodeUpdate(node_model)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (4, 6))
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([2, 2, 0, 1])

        received = gnn.segment_sum(x_edge, receivers, 3)
        expected = node_model(jnp.concatenate((x, received), axis=-1))
        result = update(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite inputs update only the destination node set."""
        node_model = nn.MLP([9, 7], key=jax.random.key(0))
        update = gnn.NodeUpdate(node_model)
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        x_edge = jax.random.normal(jax.random.key(3), (3, 4))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        received = gnn.segment_sum(x_edge, receivers, 2)
        expected = node_model(jnp.concatenate((x_dst, received), axis=-1))
        result = update((x_src, x_dst), senders, receivers, x_edge=x_edge)

        assert result.shape == (2, 7)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_segment_mean(self):
        """A compatible non-default aggregation callable controls reduction."""
        node_model = nn.MLP([10, 5], key=jax.random.key(0))
        update = gnn.NodeUpdate(node_model, aggregate=gnn.segment_mean)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (4, 6))
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([2, 2, 0, 1])

        received = gnn.segment_mean(x_edge, receivers, 3)
        expected = node_model(jnp.concatenate((x, received), axis=-1))
        result = update(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_isolated_node_receives_zero_sum(self):
        """The default aggregation gives isolated nodes a zero message."""
        update = gnn.NodeUpdate(lambda inputs: inputs[:, -2:])
        x = jax.random.normal(jax.random.key(0), (3, 4))

        result = update(
            x,
            jnp.array([0]),
            jnp.array([1]),
            x_edge=jnp.ones((1, 2)),
        )

        npt.assert_array_equal(result[2], jnp.zeros(2))

    def test_creates_no_params(self):
        """All parameters belong to the caller-supplied node model."""
        node_model = nn.MLP([10, 8, 5], key=jax.random.key(0))
        update = gnn.NodeUpdate(node_model)

        assert update.num_params == node_model.num_params

    def test_jit_and_grad(self):
        """The update composes with JIT and differentiation."""
        update = gnn.NodeUpdate(nn.MLP([10, 8, 5], key=jax.random.key(0)))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (4, 6))
        senders = jnp.array([0, 1, 2, 0])
        receivers = jnp.array([2, 2, 0, 1])

        expected = update(x, senders, receivers, x_edge=x_edge)
        result = jax.jit(update)(x, senders, receivers, x_edge=x_edge)
        grads = jax.grad(lambda model: model(x, senders, receivers, x_edge=x_edge).sum())(update)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))

    def test_bfloat16(self):
        """Reduced-precision inputs preserve their dtype and stay finite."""
        update = gnn.NodeUpdate(nn.MLP([10, 8, 5], key=jax.random.key(0))).astype(jnp.bfloat16)
        x = jax.random.normal(jax.random.key(1), (3, 4), dtype=jnp.bfloat16)
        x_edge = jax.random.normal(jax.random.key(2), (4, 6), dtype=jnp.bfloat16)
        result = update(
            x,
            jnp.array([0, 1, 2, 0]),
            jnp.array([2, 2, 0, 1]),
            x_edge=x_edge,
        )

        assert result.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(result))
