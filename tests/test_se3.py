"""SE(3) group laws, quaternion handling and the exponential map."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import se3


def _random_pose(rng):
    return se3.exp(rng.normal(size=6))


def test_quaternion_normalisation_and_sign():
    q = se3.quat_normalize(np.array([0.0, 0.0, 0.0, -2.0]))
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert q[3] > 0.0, "scalar part is forced non-negative so log is single valued"


def test_quat_multiply_matches_rotation_product(rng):
    for _ in range(30):
        a = se3.so3_exp(rng.normal(size=3))
        b = se3.so3_exp(rng.normal(size=3))
        lhs = se3.quat_to_rotation(se3.quat_multiply(a, b))
        rhs = se3.quat_to_rotation(a) @ se3.quat_to_rotation(b)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_rotation_quaternion_round_trip(rng):
    for _ in range(50):
        q = se3.so3_exp(rng.normal(size=3) * 2.0)
        R = se3.quat_to_rotation(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(se3.quat_to_rotation(se3.rotation_to_quat(R)), R, atol=1e-12)


def test_rotation_to_quat_all_branches():
    """Exercise each of Shepperd's four largest-component branches."""
    angles = [0.0, np.pi - 1e-6]
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    for ang in angles:
        for ax in axes:
            R = se3.quat_to_rotation(se3.so3_exp(ax * ang))
            q = se3.rotation_to_quat(R)
            assert np.allclose(se3.quat_to_rotation(q), R, atol=1e-9)


def test_so3_exp_log_round_trip(rng):
    for _ in range(200):
        v = rng.normal(size=3)
        phi = v / np.linalg.norm(v) * rng.uniform(0.0, np.pi - 1e-9)
        assert np.allclose(se3.so3_log(se3.so3_exp(phi)), phi, atol=1e-10)


def test_so3_near_zero_rotation():
    for eps in (0.0, 1e-15, 1e-12, 1e-9, 1e-6):
        phi = np.array([eps, -eps, 2.0 * eps])
        out = se3.so3_log(se3.so3_exp(phi))
        assert np.all(np.isfinite(out))
        assert np.allclose(out, phi, atol=1e-15 + 1e-9 * eps)


def test_so3_near_pi_rotation():
    for eps in (1e-3, 1e-6, 1e-9):
        phi = np.array([np.pi - eps, 0.0, 0.0])
        assert np.allclose(se3.so3_log(se3.so3_exp(phi)), phi, atol=1e-7)


def test_se3_exp_log_round_trip(rng):
    for _ in range(200):
        v = rng.normal(size=3)
        phi = v / np.linalg.norm(v) * rng.uniform(0.0, np.pi - 1e-9)
        xi = np.concatenate([rng.normal(size=3) * 3.0, phi])
        assert np.allclose(se3.log(se3.exp(xi)), xi, atol=1e-9)


def test_se3_exp_pure_translation():
    xi = np.array([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
    T = se3.exp(xi)
    assert np.allclose(T[:3], xi[:3], atol=1e-14)
    assert np.allclose(T[3:], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-14)


def test_inverse_and_compose(rng):
    for _ in range(50):
        X = _random_pose(rng)
        ident = se3.compose(X, se3.inverse(X))
        assert np.allclose(ident[:3], 0.0, atol=1e-12)
        assert np.allclose(np.abs(ident[6]), 1.0, atol=1e-12)


def test_compose_matches_homogeneous_matrices(rng):
    for _ in range(50):
        A, B = _random_pose(rng), _random_pose(rng)
        lhs = se3.to_matrix(se3.compose(A, B))
        rhs = se3.to_matrix(A) @ se3.to_matrix(B)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_matrix_round_trip(rng):
    X = np.atleast_2d(se3.exp(rng.normal(size=(25, 6))))
    back = se3.from_matrix(se3.to_matrix(X))
    assert np.allclose(back, X, atol=1e-11)


def test_repeated_composition_keeps_unit_quaternion(rng):
    """Renormalising on every product is what stops long chains drifting."""
    X = se3.identity()
    step = se3.exp(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.05]))
    for _ in range(5000):
        X = se3.compose(X, step)
    assert np.isclose(np.linalg.norm(X[3:]), 1.0, atol=1e-12)


def test_adjoint_definition(rng):
    for _ in range(30):
        X = _random_pose(rng)
        d = rng.normal(size=6) * 1e-5
        lhs = se3.exp(se3.adjoint(X) @ d)
        rhs = se3.compose(se3.compose(X, se3.exp(d)), se3.inverse(X))
        assert np.max(np.abs(se3.log(se3.between(rhs, lhs)))) < 1e-9


def test_left_jacobian_definition(rng):
    for _ in range(30):
        xi = rng.normal(size=6)
        Jl = se3.left_jacobian(xi)
        for k in range(6):
            d = np.zeros(6)
            d[k] = 1e-6
            lhs = se3.exp(xi + d)
            rhs = se3.compose(se3.exp(Jl @ d), se3.exp(xi))
            assert np.max(np.abs(se3.log(se3.between(rhs, lhs)))) < 1e-10


def test_se3_left_jacobian_reduces_to_so3_for_pure_rotation(rng):
    """With zero translation the SE(3) Jacobian must contain the SO(3) one twice."""
    for _ in range(20):
        phi = rng.normal(size=3)
        xi = np.concatenate([np.zeros(3), phi])
        J = se3.left_jacobian(xi)
        Jso3 = se3.so3_left_jacobian(phi)
        assert np.allclose(J[:3, :3], Jso3, atol=1e-11)
        assert np.allclose(J[3:, 3:], Jso3, atol=1e-11)
        assert np.allclose(J[3:, :3], 0.0, atol=1e-13)


def test_so3_jacobian_inverses(rng):
    for _ in range(30):
        phi = rng.normal(size=3)
        assert np.allclose(
            se3.so3_left_jacobian(phi) @ se3.so3_left_jacobian_inv(phi), np.eye(3), atol=1e-11
        )


def test_skew_is_cross_product(rng):
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(se3.skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_plus_minus_are_inverses(rng):
    X = _random_pose(rng)
    d = rng.normal(size=6) * 0.2
    assert np.allclose(se3.minus(se3.plus(X, d), X), d, atol=1e-11)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        se3.exp(np.zeros(7))
    with pytest.raises(ValueError):
        se3.log(np.zeros(6))
