"""The synthetic front end: does it produce a graph worth solving?"""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import analysis, frontend_stub
from posegraph.solver import SolverOptions, optimize


def test_se2_simulation_shape_and_loops():
    sim = frontend_stub.simulate_se2(num_poses=120, turn_every=10, min_index_gap=25, seed=0)
    assert sim.graph.num_poses == 120
    assert sim.ground_truth.shape == (120, 3)
    assert sim.num_loop_closures > 0
    assert sim.graph.num_edges == 119 + sim.num_loop_closures
    assert sim.graph.fixed == [0]


def test_se2_simulation_is_deterministic():
    a = frontend_stub.simulate_se2(num_poses=50, seed=42)
    b = frontend_stub.simulate_se2(num_poses=50, seed=42)
    assert np.allclose(a.graph.poses, b.graph.poses, atol=0.0)
    c = frontend_stub.simulate_se2(num_poses=50, seed=43)
    assert not np.allclose(a.graph.poses, c.graph.poses)


def test_odometry_estimate_is_the_chained_measurement():
    """The initial guess must have zero odometry residual, like a real front end."""
    sim = frontend_stub.simulate_se2(num_poses=60, turn_every=10, seed=1)
    chi = sim.graph.edge_chi2()
    tags = np.asarray(sim.graph.edge_tag)
    assert chi[tags == "odometry"].max() < 1e-18
    assert chi[tags == "loop"].max() > 0.0, "loop closures are what create the conflict"


def test_optimisation_improves_the_simulated_trajectory():
    sim = frontend_stub.simulate_se2(num_poses=150, turn_every=10, min_index_gap=25, seed=4)
    before = analysis.absolute_trajectory_error(sim.graph.poses, sim.ground_truth, "SE2")
    optimize(sim.graph, SolverOptions(method="lm", max_iterations=60))
    after = analysis.absolute_trajectory_error(sim.graph.poses, sim.ground_truth, "SE2")
    assert after.rmse < before.rmse


def test_se3_simulation_produces_unit_quaternions():
    sim = frontend_stub.simulate_se3(num_poses=80, seed=0)
    assert np.allclose(np.linalg.norm(sim.graph.poses[:, 3:], axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(sim.ground_truth[:, 3:], axis=1), 1.0, atol=1e-12)
    assert sim.graph.space == "SE3"


def test_false_loop_closures_are_added_and_indexed():
    sim = frontend_stub.simulate_se2(num_poses=100, turn_every=10, seed=2)
    n0 = sim.graph.num_edges
    bad, idx = frontend_stub.inject_false_loop_closures(sim.graph, 7, seed=2, min_index_gap=25)
    assert bad.num_edges == n0 + 7
    assert len(idx) == 7
    assert all(bad.edge_tag[k] == "false_loop" for k in idx)
    assert sim.graph.num_edges == n0, "the original graph must not be modified"


def test_false_loop_closures_raise_the_cost():
    sim = frontend_stub.simulate_se2(num_poses=100, turn_every=10, seed=5)
    bad, idx = frontend_stub.inject_false_loop_closures(sim.graph, 5, seed=5, min_index_gap=25)
    assert bad.chi2() > sim.graph.chi2()
    assert bad.edge_chi2()[idx].min() > np.median(bad.edge_chi2())


def test_inject_rejects_a_graph_that_is_too_small():
    sim = frontend_stub.simulate_se2(num_poses=10, seed=0)
    with pytest.raises(ValueError):
        frontend_stub.inject_false_loop_closures(sim.graph, 1, min_index_gap=50)


def test_perturb_leaves_fixed_poses_alone():
    sim = frontend_stub.simulate_se2(num_poses=40, seed=0)
    bad = frontend_stub.perturb_poses(sim.graph, sigma=0.3, seed=0)
    assert np.allclose(bad.pose(0), sim.graph.pose(0), atol=0.0)
    assert not np.allclose(bad.poses, sim.graph.poses)


def test_se3_false_loop_closures():
    sim = frontend_stub.simulate_se3(num_poses=80, seed=1)
    bad, idx = frontend_stub.inject_false_loop_closures(sim.graph, 4, seed=1, min_index_gap=20)
    assert len(idx) == 4
    assert np.allclose(np.linalg.norm(bad.edge_z[:, 3:], axis=1), 1.0, atol=1e-9)
