"""SE(2) Lie group machinery with analytic Jacobians.

Conventions used throughout this package
----------------------------------------
* A pose is stored as ``(x, y, theta)`` in a float array of shape ``(..., 3)``.
* A tangent vector is ordered ``(dx, dy, dtheta)`` -- translation first.
* The retraction is a *right* perturbation: ``X [+] d = X * Exp(d)`` and the
  local difference is ``X [-] Y = Log(Y^-1 * X)``.
* Every routine is vectorised: pass a single pose of shape ``(3,)`` or a stack
  of shape ``(N, 3)``. The leading dimensions are preserved.

The right Jacobian of SE(2) is written in closed form using the fact that the
little adjoint ``ad`` of ``se(2)`` satisfies ``ad^3 = -theta^2 ad``::

    Jl(xi) = I + B(theta) * ad + C(theta) * ad^2
    B(theta) = (1 - cos theta) / theta^2
    C(theta) = (theta - sin theta) / theta^3

Both coefficients are replaced by their Taylor expansions below
``SMALL_ANGLE`` so the formulas stay accurate (and differentiable) at
``theta -> 0``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "DOF",
    "SMALL_ANGLE",
    "identity",
    "normalize_angle",
    "compose",
    "inverse",
    "between",
    "exp",
    "log",
    "adjoint",
    "left_jacobian",
    "left_jacobian_inv",
    "right_jacobian",
    "right_jacobian_inv",
    "plus",
    "minus",
    "to_matrix",
    "from_matrix",
    "relative_error",
    "relative_error_jacobians",
]

#: Degrees of freedom of SE(2).
DOF = 3

#: Rotation magnitude below which the Taylor branch of the series coefficients
#: is used instead of the trigonometric form. Chosen so that both branches are
#: accurate to better than 1e-11 relative *at* the threshold: above it the
#: trigonometric form still has no damaging cancellation, below it the
#: truncated series is exact to double precision.
SMALL_ANGLE = 1e-2


def _as_batch(x: np.ndarray, width: int) -> Tuple[np.ndarray, bool]:
    """Return ``(batched_view, was_single)`` for an array of trailing size ``width``."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise ValueError(f"expected trailing dimension {width}, got {arr.shape}")
        return arr.reshape(1, width), True
    if arr.shape[-1] != width:
        raise ValueError(f"expected trailing dimension {width}, got {arr.shape}")
    return arr.reshape(-1, width), False


def _unbatch(arr: np.ndarray, was_single: bool) -> np.ndarray:
    return arr[0] if was_single else arr


def normalize_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap an angle (or array of angles) into ``(-pi, pi]``."""
    t = np.asarray(theta, dtype=float)
    return -((-t + np.pi) % (2.0 * np.pi) - np.pi)


def identity(n: int | None = None) -> np.ndarray:
    """Identity pose, or a stack of ``n`` identity poses."""
    if n is None:
        return np.zeros(3)
    return np.zeros((n, 3))


def _coeff_B(theta: np.ndarray) -> np.ndarray:
    """``(1 - cos t) / t^2`` with a Taylor branch near zero."""
    t2 = theta * theta
    small = np.abs(theta) < SMALL_ANGLE
    safe = np.where(small, 1.0, t2)
    big = (1.0 - np.cos(theta)) / safe
    ser = 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0
    return np.where(small, ser, big)


def _coeff_C(theta: np.ndarray) -> np.ndarray:
    """``(t - sin t) / t^3`` with a Taylor branch near zero."""
    t2 = theta * theta
    small = np.abs(theta) < SMALL_ANGLE
    safe = np.where(small, 1.0, t2 * theta)
    big = (theta - np.sin(theta)) / safe
    ser = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362880.0
    return np.where(small, ser, big)


def _rotation(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.empty(theta.shape + (2, 2))
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Group product ``a * b``."""
    A, sa = _as_batch(a, 3)
    B, sb = _as_batch(b, 3)
    c, s = np.cos(A[:, 2]), np.sin(A[:, 2])
    out = np.empty((max(A.shape[0], B.shape[0]), 3))
    out[:, 0] = A[:, 0] + c * B[:, 0] - s * B[:, 1]
    out[:, 1] = A[:, 1] + s * B[:, 0] + c * B[:, 1]
    out[:, 2] = normalize_angle(A[:, 2] + B[:, 2])
    return _unbatch(out, sa and sb)


def inverse(a: np.ndarray) -> np.ndarray:
    """Group inverse ``a^-1``."""
    A, single = _as_batch(a, 3)
    c, s = np.cos(A[:, 2]), np.sin(A[:, 2])
    out = np.empty_like(A)
    out[:, 0] = -(c * A[:, 0] + s * A[:, 1])
    out[:, 1] = -(-s * A[:, 0] + c * A[:, 1])
    out[:, 2] = normalize_angle(-A[:, 2])
    return _unbatch(out, single)


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative pose ``a^-1 * b``."""
    return compose(inverse(a), b)


def exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map ``se(2) -> SE(2)``."""
    X, single = _as_batch(xi, 3)
    theta = X[:, 2]
    A = _sinc(theta)
    B = _coeff_B(theta)
    out = np.empty_like(X)
    out[:, 0] = A * X[:, 0] - B * theta * X[:, 1]
    out[:, 1] = B * theta * X[:, 0] + A * X[:, 1]
    out[:, 2] = normalize_angle(theta)
    return _unbatch(out, single)


def _sinc(theta: np.ndarray) -> np.ndarray:
    """``sin(t)/t`` with a Taylor branch near zero."""
    t2 = theta * theta
    small = np.abs(theta) < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    big = np.sin(theta) / safe
    ser = 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0
    return np.where(small, ser, big)


def log(X: np.ndarray) -> np.ndarray:
    """Logarithm map ``SE(2) -> se(2)``."""
    P, single = _as_batch(X, 3)
    theta = normalize_angle(P[:, 2])
    A = _sinc(theta)
    B = _coeff_B(theta)
    det = A * A + (B * theta) ** 2
    # V^-1 = 1/det * [[A, B*t], [-B*t, A]]
    out = np.empty_like(P)
    out[:, 0] = (A * P[:, 0] + B * theta * P[:, 1]) / det
    out[:, 1] = (-B * theta * P[:, 0] + A * P[:, 1]) / det
    out[:, 2] = theta
    return _unbatch(out, single)


def adjoint(X: np.ndarray) -> np.ndarray:
    """Adjoint matrix ``Ad_X`` of shape ``(..., 3, 3)``."""
    P, single = _as_batch(X, 3)
    n = P.shape[0]
    Ad = np.zeros((n, 3, 3))
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    Ad[:, 0, 0] = c
    Ad[:, 0, 1] = -s
    Ad[:, 1, 0] = s
    Ad[:, 1, 1] = c
    Ad[:, 0, 2] = P[:, 1]
    Ad[:, 1, 2] = -P[:, 0]
    Ad[:, 2, 2] = 1.0
    return _unbatch(Ad, single)


def _little_adjoint(xi: np.ndarray) -> np.ndarray:
    """``ad_xi`` of shape ``(N, 3, 3)`` for a batch of tangent vectors."""
    n = xi.shape[0]
    ad = np.zeros((n, 3, 3))
    ad[:, 0, 1] = -xi[:, 2]
    ad[:, 1, 0] = xi[:, 2]
    ad[:, 0, 2] = xi[:, 1]
    ad[:, 1, 2] = -xi[:, 0]
    return ad


def left_jacobian(xi: np.ndarray) -> np.ndarray:
    """Left Jacobian ``Jl(xi)`` of SE(2), shape ``(..., 3, 3)``."""
    X, single = _as_batch(xi, 3)
    ad = _little_adjoint(X)
    B = _coeff_B(X[:, 2])[:, None, None]
    C = _coeff_C(X[:, 2])[:, None, None]
    J = np.eye(3)[None, :, :] + B * ad + C * (ad @ ad)
    return _unbatch(J, single)


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(xi) = Jl(-xi)``."""
    X, single = _as_batch(xi, 3)
    J = left_jacobian(-X)
    return _unbatch(np.atleast_3d(J).reshape(-1, 3, 3), single)


def left_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian."""
    X, single = _as_batch(xi, 3)
    J = np.linalg.inv(left_jacobian(X).reshape(-1, 3, 3))
    return _unbatch(J, single)


def right_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian."""
    X, single = _as_batch(xi, 3)
    J = np.linalg.inv(right_jacobian(X).reshape(-1, 3, 3))
    return _unbatch(J, single)


def plus(X: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Box-plus retraction ``X * Exp(delta)``."""
    return compose(X, exp(delta))


def minus(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Box-minus ``Log(Y^-1 * X)``."""
    return log(between(Y, X))


def to_matrix(X: np.ndarray) -> np.ndarray:
    """Homogeneous 3x3 matrix form of a pose."""
    P, single = _as_batch(X, 3)
    n = P.shape[0]
    M = np.zeros((n, 3, 3))
    M[:, :2, :2] = _rotation(P[:, 2])
    M[:, 0, 2] = P[:, 0]
    M[:, 1, 2] = P[:, 1]
    M[:, 2, 2] = 1.0
    return _unbatch(M, single)


def from_matrix(M: np.ndarray) -> np.ndarray:
    """Pose vector from a homogeneous 3x3 matrix."""
    A = np.asarray(M, dtype=float)
    single = A.ndim == 2
    A = A.reshape(-1, 3, 3)
    out = np.empty((A.shape[0], 3))
    out[:, 0] = A[:, 0, 2]
    out[:, 1] = A[:, 1, 2]
    out[:, 2] = np.arctan2(A[:, 1, 0], A[:, 0, 0])
    return _unbatch(out, single)


def relative_error(Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Error ``Log(Z^-1 * Xi^-1 * Xj)`` of a binary relative-pose measurement."""
    return log(between(Z, between(Xi, Xj)))


def relative_error_jacobians(
    Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic error and Jacobians of a binary SE(2) edge.

    Returns ``(e, Ji, Jj)`` where ``Ji = de/d(delta_i)``, ``Jj = de/d(delta_j)``
    under the right perturbation ``X [+] d = X * Exp(d)``.

    The closed forms are::

        e  = Log(Z^-1 Xi^-1 Xj)
        Ji = -Jr^-1(e) * Ad(Xj^-1 Xi)
        Jj =  Jr^-1(e)
    """
    Pi, si = _as_batch(Xi, 3)
    Pj, sj = _as_batch(Xj, 3)
    Pz, sz = _as_batch(Z, 3)
    e = log(between(Pz, between(Pi, Pj)))
    e = np.atleast_2d(e)
    Jri = right_jacobian_inv(e).reshape(-1, 3, 3)
    Ad = adjoint(between(Pj, Pi)).reshape(-1, 3, 3)
    Jj = Jri
    Ji = -Jri @ Ad
    single = si and sj and sz
    return _unbatch(e, single), _unbatch(Ji, single), _unbatch(Jj, single)
