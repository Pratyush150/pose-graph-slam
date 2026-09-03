"""Diagnostics and trajectory error, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import analysis, frontend_stub, se2
from posegraph.graph import PoseGraph


def test_umeyama_recovers_a_known_rotation_and_translation():
    ang = 0.7
    R_true = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    t_true = np.array([3.0, -1.5])
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    dst = (R_true @ src.T).T + t_true
    R, t, s = analysis.umeyama(src, dst)
    assert np.allclose(R, R_true, atol=1e-12)
    assert np.allclose(t, t_true, atol=1e-12)
    assert np.isclose(s, 1.0)


def test_umeyama_recovers_scale_when_asked():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    dst = 2.5 * src + np.array([1.0, 2.0, 3.0])
    R, t, s = analysis.umeyama(src, dst, with_scale=True)
    assert np.isclose(s, 2.5, atol=1e-9)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_umeyama_never_returns_a_reflection():
    """A mirrored point set must produce a rotation, not a determinant of -1."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dst = src * np.array([1.0, -1.0])
    R, _t, _s = analysis.umeyama(src, dst)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_umeyama_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        analysis.umeyama(np.zeros((3, 2)), np.zeros((4, 2)))


def test_ate_is_zero_for_an_identical_trajectory():
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.3], [2.0, 1.0, 0.6]])
    err = analysis.absolute_trajectory_error(poses, poses, "SE2")
    assert err.rmse < 1e-12
    assert err.n == 3


def test_ate_is_invariant_to_a_rigid_transform():
    """This is the whole reason ATE aligns first: the gauge is free."""
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5], [0.0, 1.0, 1.0]])
    moved = np.atleast_2d(se2.compose(np.array([5.0, -2.0, 1.1]), poses))
    err = analysis.absolute_trajectory_error(moved, poses, "SE2")
    assert err.rmse < 1e-10
    unaligned = analysis.absolute_trajectory_error(moved, poses, "SE2", align=False)
    assert unaligned.rmse > 1.0


def test_ate_matches_a_hand_computed_offset():
    """Two poses pushed apart by a known amount, in a case alignment cannot absorb."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    est = np.array([[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.0, 0.0]])
    # after alignment the middle point keeps a residual; check against a direct
    # evaluation of the same definition
    R, t, s = analysis.umeyama(est[:, :2], gt[:, :2])
    aligned = (s * (R @ est[:, :2].T)).T + t
    expected = np.sqrt(np.mean(np.sum((aligned - gt[:, :2]) ** 2, axis=1)))
    err = analysis.absolute_trajectory_error(est, gt, "SE2")
    assert np.isclose(err.rmse, expected, rtol=1e-12)
    assert err.max >= err.mean


def test_rpe_is_zero_for_an_identical_trajectory():
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.3], [2.0, 1.0, 0.6], [3.0, 1.0, 0.9]])
    err = analysis.relative_pose_error(poses, poses, "SE2", delta=1)
    assert err.rmse < 1e-12
    assert err.rotation_rmse_deg is not None and err.rotation_rmse_deg < 1e-9


def test_rpe_matches_a_hand_computed_step_error():
    """One estimated step is 0.1 m too long; RPE(delta=1) must see exactly that."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    est = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])
    err = analysis.relative_pose_error(est, gt, "SE2", delta=1)
    assert np.isclose(err.max, 0.1, atol=1e-12)
    assert np.isclose(err.rmse, 0.1 / np.sqrt(2.0) * np.sqrt(1.0), atol=1e-12) or np.isclose(
        err.rmse, np.sqrt((0.1**2 + 0.0**2) / 2.0), atol=1e-12
    )


def test_rpe_is_unaffected_by_a_global_transform():
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [2.0, 0.5, 0.4]])
    est = np.atleast_2d(se2.compose(np.array([7.0, 3.0, 2.0]), gt))
    err = analysis.relative_pose_error(est, gt, "SE2", delta=1)
    assert err.rmse < 1e-12


def test_rpe_rejects_bad_delta():
    poses = np.zeros((3, 3))
    with pytest.raises(ValueError):
        analysis.relative_pose_error(poses, poses, "SE2", delta=0)
    with pytest.raises(ValueError):
        analysis.relative_pose_error(poses, poses, "SE2", delta=5)


def test_se3_ate_and_rpe_run():
    sim = frontend_stub.simulate_se3(num_poses=40, seed=1)
    ate = analysis.absolute_trajectory_error(sim.graph.poses, sim.ground_truth, "SE3")
    rpe = analysis.relative_pose_error(sim.graph.poses, sim.ground_truth, "SE3", delta=1)
    assert ate.n == 40 and rpe.n == 39
    assert ate.rmse > 0.0
    assert "rmse" in ate.describe() and "rpe" in rpe.describe()


def test_chi2_report_splits_by_edge_class():
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.zeros(3))
    for k in range(1, 5):
        z = np.array([1.0, 0.0, 0.0])
        g.add_pose(k, se2.compose(g.pose(k - 1), z))
        g.add_edge(k - 1, k, z, np.eye(3))
    g.add_edge(0, 4, np.array([3.0, 0.0, 0.0]), np.eye(3), tag="loop")
    rep = analysis.chi2_report(g)
    assert np.isclose(rep.odometry, 0.0, atol=1e-12)
    assert np.isclose(rep.loop, 1.0, atol=1e-12), "the loop edge is 1 m short"
    assert np.isclose(rep.total, 1.0, atol=1e-12)
    assert rep.dof >= 1
    assert "chi2/dof" in rep.describe()


def test_edge_residuals_and_ranking():
    g = PoseGraph(space="SE2")
    for k in range(4):
        g.add_pose(k, np.array([float(k), 0.0, 0.0]))
    for k in range(1, 4):
        g.add_edge(k - 1, k, np.array([1.0, 0.0, 0.0]), np.eye(3))
    g.add_edge(0, 3, np.array([9.0, 0.0, 0.0]), np.eye(3), tag="loop")
    res = analysis.edge_residuals(g)
    assert len(res) == 4
    worst = analysis.rank_outliers(g, top=1)[0]
    assert worst.tag == "loop"
    assert np.isclose(worst.chi2, 36.0, atol=1e-9)
    assert np.isclose(worst.mahalanobis, 6.0, atol=1e-9)


def test_rank_outliers_respects_top():
    sim = frontend_stub.simulate_se2(num_poses=60, turn_every=10, seed=2)
    assert len(analysis.rank_outliers(sim.graph, top=5)) == 5
    assert analysis.rank_outliers(sim.graph, top=0) == []
