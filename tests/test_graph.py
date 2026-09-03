"""Graph container: construction, residuals, topology and spanning-tree init."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import se2, se3
from posegraph.graph import PoseGraph


def _chain(n=6):
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.zeros(3))
    for k in range(1, n):
        z = np.array([1.0, 0.0, 0.3])
        g.add_pose(k, se2.compose(g.pose(k - 1), z))
        g.add_edge(k - 1, k, z, np.eye(3) * 4.0)
    return g


def test_construction_counts():
    g = _chain(6)
    assert g.num_poses == 6
    assert g.num_edges == 5
    assert g.dim == 3
    assert g.point_dim == 2
    assert "PoseGraph(SE2)" in g.summary()


def test_perfect_chain_has_zero_chi2():
    g = _chain(8)
    assert g.chi2() < 1e-20
    assert np.allclose(g.edge_chi2(), 0.0, atol=1e-20)


def test_chi2_scales_with_information():
    g = _chain(3)
    g.set_pose(2, se2.plus(g.pose(2), np.array([0.1, 0.0, 0.0])))
    base = g.chi2()
    g.edge_info *= 9.0
    assert np.isclose(g.chi2(), 9.0 * base)


def test_edge_tags_default_to_odometry_or_loop():
    g = _chain(6)
    g.add_edge(0, 5, np.zeros(3), np.eye(3))
    assert g.edge_tag[:5] == ["odometry"] * 5
    assert g.edge_tag[-1] == "loop"
    assert list(g.loop_edges()) == [5]


def test_fix_and_prior():
    g = _chain(4)
    g.fix_pose(2)
    assert g.fixed == [2]
    g.fix_pose(2)
    assert g.fixed == [2], "fixing twice must not duplicate"
    with pytest.raises(KeyError):
        g.fix_pose(99)
    g.add_prior(0, np.zeros(3), np.eye(3) * 10.0)
    assert len(g.priors) == 1
    assert len(g.prior_errors()) == 1


def test_prior_error_is_zero_at_the_measurement():
    g = _chain(3)
    g.add_prior(1, g.pose(1), np.eye(3))
    assert np.allclose(g.prior_errors()[0], 0.0, atol=1e-12)


def test_copy_is_deep():
    g = _chain(4)
    h = g.copy()
    h.set_pose(1, np.array([9.0, 9.0, 0.0]))
    h.edge_info[0] *= 2.0
    assert not np.allclose(g.pose(1), h.pose(1))
    assert not np.allclose(g.edge_info[0], h.edge_info[0])


def test_connected_components():
    g = _chain(4)
    g.add_pose(10, np.zeros(3))
    g.add_pose(11, np.zeros(3))
    g.add_edge(10, 11, np.zeros(3), np.eye(3))
    comps = g.connected_components()
    assert len(comps) == 2
    assert sorted(len(c) for c in comps) == [2, 4]


def test_spanning_tree_initialisation_recovers_the_chain():
    g = _chain(10)
    truth = g.poses.copy()
    g.poses = np.zeros_like(g.poses)
    g.initialize_from_spanning_tree()
    assert np.allclose(g.poses, truth, atol=1e-12)
    assert g.chi2() < 1e-20


def test_spanning_tree_notes_unreachable_poses():
    g = _chain(4)
    g.add_pose(50, np.zeros(3))
    g.initialize_from_spanning_tree()
    assert any("not reachable" in line for line in g.trailing_lines)


def test_se3_graph_residuals(rng):
    g = PoseGraph(space="SE3")
    g.add_pose(0, se3.identity())
    z = se3.exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2]))
    g.add_pose(1, se3.compose(g.pose(0), z))
    g.add_edge(0, 1, z, np.eye(6))
    assert g.dim == 6
    assert g.point_dim == 3
    assert g.chi2() < 1e-20
    g.set_pose(1, se3.plus(g.pose(1), np.array([0.1, 0, 0, 0, 0, 0])))
    assert g.chi2() > 1e-3


def test_landmark_edge_errors():
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.array([0.0, 0.0, np.pi / 2]))
    g.add_point(100, np.array([0.0, 2.0]))
    # the point is 2 m ahead in the pose frame after the 90-degree rotation
    g.add_landmark_edge(0, 100, np.array([2.0, 0.0]), np.eye(2))
    assert np.allclose(g.landmark_errors()[0], 0.0, atol=1e-12)
    assert g.chi2() < 1e-20


def test_bad_shapes_are_rejected():
    g = PoseGraph(space="SE2")
    with pytest.raises(ValueError):
        g.add_pose(0, np.zeros(4))
    g.add_pose(0, np.zeros(3))
    with pytest.raises(ValueError):
        g.add_edge(0, 1, np.zeros(2), np.eye(3))
    with pytest.raises(ValueError):
        g.add_edge(0, 1, np.zeros(3), np.eye(4))
    with pytest.raises(ValueError):
        PoseGraph(space="SE9")
