"""The series coefficients must be continuous where the Taylor branch switches on.

A discontinuity here is the classic silent SLAM bug: the solver takes a step,
the rotation happens to land near the threshold, the Jacobian jumps, and the
optimiser either stalls or oscillates for no visible reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import se2, se3

TINY = 1e-13


def _axis():
    a = np.array([0.31, -0.52, 0.79])
    return a / np.linalg.norm(a)


@pytest.mark.parametrize(
    "fn", [se3.so3_left_jacobian, se3.so3_left_jacobian_inv, se3.so3_right_jacobian]
)
def test_so3_jacobian_branch_is_continuous(fn):
    t = se3.SMALL_ANGLE
    ax = _axis()
    below = fn(ax * (t - TINY))
    above = fn(ax * (t + TINY))
    assert np.abs(below - above).max() < 1e-10


def test_so3_jacobian_branch_agrees_over_a_window():
    """Both branches must agree, not merely meet, on either side of the threshold."""
    ax = _axis()
    t = se3.SMALL_ANGLE
    for scale in (0.5, 0.9, 1.0, 1.1, 2.0):
        phi = ax * t * scale
        J = se3.so3_left_jacobian(phi)
        Jinv = se3.so3_left_jacobian_inv(phi)
        assert np.abs(J @ Jinv - np.eye(3)).max() < 1e-12


def test_se3_left_jacobian_branch_is_continuous():
    t = se3.SMALL_ANGLE_SE3
    ax = _axis()
    rho = np.array([0.4, -0.2, 1.1])
    below = se3.left_jacobian(np.concatenate([rho, ax * (t - TINY)]))
    above = se3.left_jacobian(np.concatenate([rho, ax * (t + TINY)]))
    assert np.abs(below - above).max() < 1e-10


def test_se2_jacobian_branch_is_continuous():
    t = se2.SMALL_ANGLE
    below = se2.left_jacobian(np.array([0.7, -0.3, t - TINY]))
    above = se2.left_jacobian(np.array([0.7, -0.3, t + TINY]))
    assert np.abs(below - above).max() < 1e-10


def test_se2_exp_branch_is_continuous():
    t = se2.SMALL_ANGLE
    below = se2.exp(np.array([1.0, 2.0, t - TINY]))
    above = se2.exp(np.array([1.0, 2.0, t + TINY]))
    assert np.abs(below - above).max() < 1e-12


def test_coefficients_are_finite_at_exact_zero():
    assert np.all(np.isfinite(se3.so3_left_jacobian(np.zeros(3))))
    assert np.all(np.isfinite(se3.so3_left_jacobian_inv(np.zeros(3))))
    assert np.all(np.isfinite(se3.left_jacobian(np.zeros(6))))
    assert np.all(np.isfinite(se2.left_jacobian(np.zeros(3))))
    assert np.allclose(se3.so3_left_jacobian(np.zeros(3)), np.eye(3))
    assert np.allclose(se2.left_jacobian(np.zeros(3)), np.eye(3))


def test_so3_jacobian_stays_finite_at_pi():
    """``Jl^-1`` is written with cot(t/2) precisely so that t = pi is not a pole."""
    ax = _axis()
    J = se3.so3_left_jacobian_inv(ax * np.pi)
    assert np.all(np.isfinite(J))
    assert np.abs(J @ se3.so3_left_jacobian(ax * np.pi) - np.eye(3)).max() < 1e-10
