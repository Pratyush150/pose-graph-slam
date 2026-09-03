"""SE(2) group laws, exponential map and adjoint."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import se2


def test_identity_is_neutral(rng):
    for _ in range(20):
        X = rng.normal(size=3)
        X[2] = se2.normalize_angle(X[2])
        assert np.allclose(se2.compose(X, se2.identity()), X)
        assert np.allclose(se2.compose(se2.identity(), X), X)


def test_inverse_round_trip(rng):
    for _ in range(50):
        X = rng.normal(size=3)
        assert np.allclose(se2.compose(X, se2.inverse(X)), np.zeros(3), atol=1e-12)
        assert np.allclose(se2.compose(se2.inverse(X), X), np.zeros(3), atol=1e-12)


def test_compose_matches_homogeneous_matrices(rng):
    for _ in range(50):
        A = rng.normal(size=3)
        B = rng.normal(size=3)
        lhs = se2.to_matrix(se2.compose(A, B))
        rhs = se2.to_matrix(A) @ se2.to_matrix(B)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_matrix_round_trip(rng):
    X = rng.normal(size=(30, 3))
    X[:, 2] = se2.normalize_angle(X[:, 2])
    assert np.allclose(se2.from_matrix(se2.to_matrix(X)), X, atol=1e-12)


def test_exp_log_round_trip(rng):
    for _ in range(200):
        xi = np.array(
            [rng.normal() * 3.0, rng.normal() * 3.0, rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)]
        )
        assert np.allclose(se2.log(se2.exp(xi)), xi, atol=1e-11)


def test_exp_of_zero_is_identity():
    assert np.allclose(se2.exp(np.zeros(3)), np.zeros(3))
    assert np.allclose(se2.log(np.zeros(3)), np.zeros(3))


def test_exp_pure_translation_is_translation():
    xi = np.array([2.0, -3.0, 0.0])
    assert np.allclose(se2.exp(xi), np.array([2.0, -3.0, 0.0]), atol=1e-14)


def test_tiny_rotation_stays_finite():
    xi = np.array([1.0, 0.5, 1e-14])
    X = se2.exp(xi)
    assert np.all(np.isfinite(X))
    assert np.allclose(X[:2], xi[:2], atol=1e-12)


def test_adjoint_definition(rng):
    """``Exp(Ad_X d) = X Exp(d) X^-1`` for small ``d``."""
    for _ in range(30):
        X = rng.normal(size=3)
        d = rng.normal(size=3) * 1e-4
        lhs = se2.exp(se2.adjoint(X) @ d)
        rhs = se2.compose(se2.compose(X, se2.exp(d)), se2.inverse(X))
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_plus_minus_are_inverses(rng):
    for _ in range(30):
        X = rng.normal(size=3)
        d = rng.normal(size=3) * 0.3
        Y = se2.plus(X, d)
        assert np.allclose(se2.minus(Y, X), d, atol=1e-11)


def test_normalize_angle_wraps():
    assert np.isclose(se2.normalize_angle(3.0 * np.pi), np.pi)
    assert np.isclose(se2.normalize_angle(-3.0 * np.pi), np.pi)
    assert np.isclose(se2.normalize_angle(0.5), 0.5)


def test_batched_matches_single(rng):
    X = rng.normal(size=(17, 3))
    Y = rng.normal(size=(17, 3))
    batched = se2.compose(X, Y)
    for k in range(17):
        assert np.allclose(batched[k], se2.compose(X[k], Y[k]))


def test_left_jacobian_definition(rng):
    """``Exp(xi + d) ~ Exp(Jl(xi) d) Exp(xi)``."""
    for _ in range(30):
        xi = np.array([rng.normal(), rng.normal(), rng.uniform(-2.0, 2.0)])
        Jl = se2.left_jacobian(xi)
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-6
            lhs = se2.exp(xi + d)
            rhs = se2.compose(se2.exp(Jl @ d), se2.exp(xi))
            assert np.max(np.abs(se2.log(se2.between(rhs, lhs)))) < 1e-10


def test_right_jacobian_is_left_of_negative(rng):
    xi = rng.normal(size=3)
    assert np.allclose(se2.right_jacobian(xi), se2.left_jacobian(-xi))


def test_jacobian_inverses(rng):
    xi = rng.normal(size=3)
    assert np.allclose(se2.left_jacobian(xi) @ se2.left_jacobian_inv(xi), np.eye(3), atol=1e-12)
    assert np.allclose(se2.right_jacobian(xi) @ se2.right_jacobian_inv(xi), np.eye(3), atol=1e-12)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        se2.exp(np.zeros(4))
    with pytest.raises(ValueError):
        se2.compose(np.zeros(2), np.zeros(3))
