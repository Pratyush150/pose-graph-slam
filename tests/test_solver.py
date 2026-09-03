"""Solver behaviour: monotone descent, recovery of a known answer, gauge handling."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import frontend_stub, se2, se3
from posegraph.graph import PoseGraph
from posegraph.solver import Problem, SolverOptions, optimize


def _perturbed(space="SE2", n=85, seed=5):
    """A small circuit that actually revisits itself, so loop closures exist."""
    sim = (
        frontend_stub.simulate_se2(num_poses=n, turn_every=10, min_index_gap=25, seed=seed)
        if space == "SE2"
        else frontend_stub.simulate_se3(num_poses=n, radius=6.0, min_index_gap=15, seed=seed)
    )
    assert sim.num_loop_closures > 0, "test fixture must contain loop closures"
    return sim


def test_lm_reduces_cost_monotonically():
    sim = _perturbed()
    result = optimize(sim.graph, SolverOptions(method="lm", max_iterations=50))
    costs = [r.cost for r in result.history]
    assert costs, "the solver must actually take steps"
    for a, b in zip(costs, costs[1:]):
        assert b <= a + 1e-9, "LM rejects any step that would increase the cost"
    assert result.final_chi2 < result.initial_chi2


def test_lm_recovers_a_noise_free_graph_to_near_zero():
    """A perfect graph perturbed away from its solution must come back to it."""
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.zeros(3))
    for k in range(1, 30):
        z = np.array([1.0, 0.0, 0.2])
        g.add_pose(k, se2.compose(g.pose(k - 1), z))
        g.add_edge(k - 1, k, z, np.eye(3) * 100.0)
    g.add_edge(0, 29, se2.between(g.pose(0), g.pose(29)), np.eye(3) * 100.0)
    g.fix_pose(0)
    truth = g.poses.copy()
    bad = frontend_stub.perturb_poses(g, sigma=0.2, seed=1)
    assert bad.chi2() > 1.0
    result = optimize(bad, SolverOptions(method="lm", max_iterations=100))
    assert result.final_chi2 < 1e-12
    assert np.abs(bad.poses - truth).max() < 1e-6


def test_se3_solver_recovers_a_noise_free_graph():
    g = PoseGraph(space="SE3")
    g.add_pose(0, se3.identity())
    for k in range(1, 25):
        z = se3.exp(np.array([1.0, 0.0, 0.1, 0.02, 0.05, 0.15]))
        g.add_pose(k, se3.compose(g.pose(k - 1), z))
        g.add_edge(k - 1, k, z, np.eye(6) * 100.0)
    g.add_edge(0, 24, se3.between(g.pose(0), g.pose(24)), np.eye(6) * 100.0)
    g.fix_pose(0)
    truth = g.poses.copy()
    bad = frontend_stub.perturb_poses(g, sigma=0.1, seed=2)
    result = optimize(bad, SolverOptions(method="lm", max_iterations=100))
    assert result.final_chi2 < 1e-12
    assert np.abs(bad.poses[:, :3] - truth[:, :3]).max() < 1e-6


@pytest.mark.parametrize("method", ["gn", "lm", "dogleg"])
def test_all_methods_reach_the_same_minimum(method):
    sim = _perturbed(n=85, seed=11)
    g = sim.graph.copy()
    result = optimize(g, SolverOptions(method=method, max_iterations=200))
    assert result.final_chi2 < 0.35 * result.initial_chi2
    reference = sim.graph.copy()
    optimize(reference, SolverOptions(method="lm", max_iterations=200))
    assert abs(g.chi2() - reference.chi2()) < 1e-4 * max(1.0, reference.chi2())


def test_unfixed_graph_is_rejected_with_a_useful_message():
    sim = _perturbed(n=85)
    g = sim.graph
    g.fixed.clear()
    with pytest.raises(ValueError, match="gauge|singular|fixed"):
        optimize(g, SolverOptions())


def test_anchoring_makes_the_information_matrix_non_singular():
    """Without a fixed pose H has a null space of exactly the gauge dimension."""
    sim = _perturbed(n=85)
    g = sim.graph
    g.fixed.clear()
    problem = Problem(g)
    data, _b, _c, _cost = problem.build_system()
    H = np.zeros((problem.n_free, problem.n_free))
    np.add.at(H, (problem.H_rows, problem.H_cols), data)
    eig = np.linalg.eigvalsh(H)
    assert np.sum(np.abs(eig) < 1e-8 * max(1.0, eig.max())) == 3, "3 gauge directions in SE(2)"

    g.fix_pose(g.pose_ids[0])
    problem2 = Problem(g)
    data2, _b2, _c2, _cost2 = problem2.build_system()
    H2 = np.zeros((problem2.n_free, problem2.n_free))
    np.add.at(H2, (problem2.H_rows, problem2.H_cols), data2)
    eig2 = np.linalg.eigvalsh(H2)
    assert eig2.min() > 1e-8 * eig2.max(), "anchoring removes the null space"


def test_prior_alone_fixes_the_gauge():
    sim = _perturbed(n=85)
    g = sim.graph
    g.fixed.clear()
    g.add_prior(g.pose_ids[0], g.pose(g.pose_ids[0]), np.eye(3) * 1e6)
    result = optimize(g, SolverOptions(method="lm", max_iterations=50))
    assert result.final_chi2 < result.initial_chi2


def test_fixed_pose_does_not_move():
    sim = _perturbed(n=85)
    g = sim.graph
    before = g.pose(g.fixed[0]).copy()
    optimize(g, SolverOptions(method="lm", max_iterations=30))
    assert np.allclose(g.pose(g.fixed[0]), before, atol=0.0)


def test_history_and_result_fields_are_populated():
    sim = _perturbed(n=85)
    result = optimize(sim.graph, SolverOptions(method="lm", max_iterations=30))
    assert result.iterations == len(result.history)
    assert result.seconds > 0.0
    assert result.backend in ("dense", "scipy_splu", "numpy_sparse")
    assert result.ordering is not None
    assert len(result.chi2_history) == result.iterations
    assert "chi2" in result.summary()
    assert result.message


def test_unknown_method_raises():
    sim = _perturbed(n=85)
    with pytest.raises(ValueError):
        optimize(sim.graph, SolverOptions(method="newton"))


def test_landmark_bundle_converges():
    """Poses and points optimised together, with mixed variable block sizes."""
    rng = np.random.default_rng(4)
    g = PoseGraph(space="SE2")
    truth_poses = []
    for k in range(6):
        p = np.array([float(k), 0.0, 0.0])
        truth_poses.append(p)
        g.add_pose(k, p)
    for k in range(1, 6):
        g.add_edge(k - 1, k, np.array([1.0, 0.0, 0.0]), np.eye(3) * 100.0)
    truth_pts = {}
    for m in range(4):
        q = np.array([1.0 + m, 2.0])
        truth_pts[100 + m] = q
        g.add_point(100 + m, q + rng.normal(0, 0.2, 2))
        for k in range(6):
            local = se2.compose(se2.inverse(g.pose(k)), np.append(q, 0.0))[:2]
            g.add_landmark_edge(k, 100 + m, local, np.eye(2) * 50.0)
    g.fix_pose(0)
    result = optimize(g, SolverOptions(method="lm", max_iterations=60))
    assert result.final_chi2 < 1e-10
    for pid, q in truth_pts.items():
        assert np.abs(g.points[g._point_index[pid]] - q).max() < 1e-5


def test_pure_numpy_backend_gives_the_same_answer_as_scipy():
    """The SciPy-free path must be a real alternative, not a decoration."""
    sim = _perturbed(n=120, seed=21)
    a = sim.graph.copy()
    b = sim.graph.copy()
    ra = optimize(a, SolverOptions(method="lm", max_iterations=60, backend="numpy_sparse"))
    rb = optimize(b, SolverOptions(method="lm", max_iterations=60, backend="dense"))
    assert ra.backend == "numpy_sparse"
    assert abs(ra.final_chi2 - rb.final_chi2) < 1e-6 * max(1.0, rb.final_chi2)
    assert np.abs(a.poses[:, :2] - b.poses[:, :2]).max() < 1e-6
