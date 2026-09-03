"""The credibility test: every analytic Jacobian against central differences.

If these pass, the derivatives the solver uses are the derivatives of the cost
it actually evaluates. If they do not, everything downstream is guesswork.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import numerical_jacobian
from posegraph import se2, se3
from posegraph.graph import PoseGraph
from posegraph.solver import Problem


def test_se2_relative_error_jacobians(rng):
    worst = 0.0
    for _ in range(200):
        Xi, Xj, Z = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        _e, Ji, Jj = se2.relative_error_jacobians(Xi, Xj, Z)
        nJi = numerical_jacobian(lambda x: se2.relative_error(x, Xj, Z), Xi, se2.plus, 3)
        nJj = numerical_jacobian(lambda x: se2.relative_error(Xi, x, Z), Xj, se2.plus, 3)
        worst = max(worst, np.abs(Ji - nJi).max(), np.abs(Jj - nJj).max())
    assert worst < 1e-6, f"largest deviation from central differences was {worst}"


def test_se3_relative_error_jacobians(rng):
    worst = 0.0
    for _ in range(200):
        Xi = se3.exp(rng.normal(size=6))
        Xj = se3.exp(rng.normal(size=6))
        Z = se3.exp(rng.normal(size=6) * 0.4)
        _e, Ji, Jj = se3.relative_error_jacobians(Xi, Xj, Z)
        nJi = numerical_jacobian(lambda x: se3.relative_error(x, Xj, Z), Xi, se3.plus, 6)
        nJj = numerical_jacobian(lambda x: se3.relative_error(Xi, x, Z), Xj, se3.plus, 6)
        worst = max(worst, np.abs(Ji - nJi).max(), np.abs(Jj - nJj).max())
    assert worst < 1e-6, f"largest deviation from central differences was {worst}"


def test_se3_jacobians_with_large_rotations(rng):
    """Near-pi relative rotations are where a sloppy Jacobian falls apart."""
    worst = 0.0
    for _ in range(60):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        Xi = se3.exp(np.concatenate([rng.normal(size=3), axis * (np.pi - 1e-3)]))
        Xj = se3.exp(rng.normal(size=6))
        Z = se3.identity()
        _e, Ji, Jj = se3.relative_error_jacobians(Xi, Xj, Z)
        nJi = numerical_jacobian(lambda x: se3.relative_error(x, Xj, Z), Xi, se3.plus, 6)
        worst = max(worst, np.abs(Ji - nJi).max())
    assert worst < 1e-5


def test_se2_prior_jacobian(rng):
    """The unary prior uses ``Jr^-1(e)``; check it the same way."""
    for _ in range(50):
        X, Z = rng.normal(size=3), rng.normal(size=3)
        e = se2.log(se2.between(Z, X))
        J = se2.right_jacobian_inv(e)
        nJ = numerical_jacobian(lambda x: se2.log(se2.between(Z, x)), X, se2.plus, 3)
        assert np.abs(J - nJ).max() < 1e-6


def test_se3_prior_jacobian(rng):
    for _ in range(50):
        X = se3.exp(rng.normal(size=6))
        Z = se3.exp(rng.normal(size=6))
        e = se3.log(se3.between(Z, X))
        J = se3.right_jacobian_inv(e)
        nJ = numerical_jacobian(lambda x: se3.log(se3.between(Z, x)), X, se3.plus, 6)
        assert np.abs(J - nJ).max() < 1e-6


def _landmark_problem(space, rng):
    g = PoseGraph(space=space)
    if space == "SE2":
        g.add_pose(0, rng.normal(size=3))
        g.add_point(100, rng.normal(size=2))
        g.add_landmark_edge(0, 100, rng.normal(size=2))
    else:
        g.add_pose(0, se3.exp(rng.normal(size=6)))
        g.add_point(100, rng.normal(size=3))
        g.add_landmark_edge(0, 100, rng.normal(size=3))
    g.add_prior(0, g.pose(0), np.eye(g.dim))
    return g, Problem(g)


@pytest.mark.parametrize("space", ["SE2", "SE3"])
def test_landmark_jacobians(space, rng):
    g, problem = _landmark_problem(space, rng)
    Jp, Jq = problem._landmark_jacobians()
    ops = g.ops
    pose0 = g.pose(0)
    point0 = g.points[0].copy()

    def err_pose(x):
        g.set_pose(0, x)
        out = g.landmark_errors()[0].copy()
        g.set_pose(0, pose0)
        return out

    def err_point(p):
        g.points[0] = p
        out = g.landmark_errors()[0].copy()
        g.points[0] = point0
        return out

    nJp = numerical_jacobian(err_pose, pose0, ops.plus, g.dim)
    nJq = numerical_jacobian(err_point, point0, lambda p, d: p + d, g.point_dim)
    assert np.abs(Jp[0] - nJp).max() < 1e-6
    assert np.abs(Jq[0] - nJq).max() < 1e-6


@pytest.mark.parametrize("space", ["SE2", "SE3"])
def test_assembled_gradient_matches_numerical(space, rng):
    """The assembled ``b`` must be the negative gradient of the total cost."""
    from posegraph import frontend_stub

    sim = (
        frontend_stub.simulate_se2(num_poses=12, turn_every=4, seed=3)
        if space == "SE2"
        else frontend_stub.simulate_se3(num_poses=12, radius=3.0, seed=3)
    )
    g = sim.graph
    problem = Problem(g)
    _data, b, _chi2, _cost = problem.build_system()

    base = g.poses.copy()
    d = g.dim
    eps = 1e-6
    numeric = np.zeros_like(b)
    for k in range(b.size):
        var = problem.free_scalar[k] // d
        comp = problem.free_scalar[k] % d
        for sign, slot in ((1.0, 0), (-1.0, 1)):
            delta = np.zeros((g.num_poses, d))
            delta[var, comp] = sign * eps
            g.poses = np.atleast_2d(g.ops.plus(base, delta))
            c = problem.total_cost()[1]
            numeric[k] += -sign * c / (2.0 * eps)
        g.poses = base.copy()
    # b = -0.5 * dF/dx because F = sum e^T Omega e and b = -J^T Omega e
    assert np.abs(numeric * 0.5 - b).max() < 1e-4 * max(1.0, np.abs(b).max())
