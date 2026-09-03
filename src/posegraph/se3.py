"""SE(3) Lie group machinery with analytic Jacobians, quaternion based.

Conventions
-----------
* A pose is ``(x, y, z, qx, qy, qz, qw)`` in an array of shape ``(..., 7)``.
  The quaternion is Hamilton, scalar-last, and is renormalised after every
  product so repeated composition cannot drift off the unit sphere.
* A tangent vector is ``(dx, dy, dz, wx, wy, wz)`` -- translation first, then
  the rotation vector. This matches the ordering that ``EDGE_SE3:QUAT``
  information matrices use in the ``.g2o`` format.
* Retraction is a right perturbation: ``X [+] d = X * Exp(d)``.

The left Jacobian of SE(3) is built from the closed form of the series
``Jl = sum_n ad^n / (n+1)!``. Because the little adjoint of ``se(3)`` satisfies
``ad^5 = -2 t^2 ad^3 - t^4 ad`` (with ``t = |phi|``) the series collapses to a
quartic polynomial in ``ad``::

    Jl = I + a ad + b ad^2 + c ad^3 + d ad^4
    a = (4 - t sin t - 4 cos t) / (2 t^2)
    b = (4 t - 5 sin t + t cos t) / (2 t^3)
    c = (2 - t sin t - 2 cos t) / (2 t^4)
    d = (2 t - 3 sin t + t cos t) / (2 t^5)

Each coefficient switches to its Taylor expansion below ``SMALL_ANGLE``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "DOF",
    "SMALL_ANGLE",
    "SMALL_ANGLE_SE3",
    "SMALL_NORM",
    "identity",
    "skew",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_to_rotation",
    "rotation_to_quat",
    "so3_exp",
    "so3_log",
    "so3_left_jacobian",
    "so3_left_jacobian_inv",
    "so3_right_jacobian",
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
    "from_rt",
    "translation",
    "relative_error",
    "relative_error_jacobians",
]

#: Degrees of freedom of SE(3).
DOF = 6

#: Rotation magnitude below which the Taylor branches of the SO(3) series
#: coefficients are used. Both branches agree to ~1e-11 relative at the
#: threshold; see ``tests/test_small_angle_branches.py``.
SMALL_ANGLE = 1e-2

#: Threshold for the SE(3) left-Jacobian polynomial coefficients. It is larger
#: than ``SMALL_ANGLE`` because the ``d`` coefficient divides by ``t^5`` and
#: loses roughly five digits to cancellation before that.
SMALL_ANGLE_SE3 = 1e-1

#: Threshold below which the quaternion logarithm uses its series branch.
SMALL_NORM = 1e-10


def _as_batch(x: np.ndarray, width: int) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise ValueError(f"expected trailing dimension {width}, got {arr.shape}")
        return arr.reshape(1, width), True
    if arr.shape[-1] != width:
        raise ValueError(f"expected trailing dimension {width}, got {arr.shape}")
    return arr.reshape(-1, width), False


def _unbatch(arr: np.ndarray, single: bool) -> np.ndarray:
    return arr[0] if single else arr


def identity(n: int | None = None) -> np.ndarray:
    """Identity pose, or a stack of ``n`` identity poses."""
    if n is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = np.zeros((n, 7))
    out[:, 6] = 1.0
    return out


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector (batched)."""
    V = np.asarray(v, dtype=float)
    single = V.ndim == 1
    V = V.reshape(-1, 3)
    n = V.shape[0]
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -V[:, 2]
    S[:, 0, 2] = V[:, 1]
    S[:, 1, 0] = V[:, 2]
    S[:, 1, 2] = -V[:, 0]
    S[:, 2, 0] = -V[:, 1]
    S[:, 2, 1] = V[:, 0]
    return _unbatch(S, single)


# --------------------------------------------------------------------------
# quaternion helpers
# --------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalise a quaternion and force a non-negative scalar part."""
    Q, single = _as_batch(q, 4)
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    out = Q / n
    sign = np.where(out[:, 3:4] < 0.0, -1.0, 1.0)
    return _unbatch(out * sign, single)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for scalar-last quaternions."""
    A, sa = _as_batch(a, 4)
    B, sb = _as_batch(b, 4)
    ax, ay, az, aw = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    bx, by, bz, bw = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    n = max(A.shape[0], B.shape[0])
    out = np.empty((n, 4))
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    return _unbatch(out, sa and sb)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions)."""
    Q, single = _as_batch(q, 4)
    out = Q.copy()
    out[:, :3] *= -1.0
    return _unbatch(out, single)


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrices of shape ``(..., 3, 3)`` from unit quaternions."""
    Q, single = _as_batch(q, 4)
    Q = np.asarray(quat_normalize(Q)).reshape(-1, 4)
    x, y, z, w = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    n = Q.shape[0]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return _unbatch(R, single)


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion from a rotation matrix, via the numerically stable
    largest-component branch (Shepperd's method)."""
    A = np.asarray(R, dtype=float)
    single = A.ndim == 2
    A = A.reshape(-1, 3, 3)
    n = A.shape[0]
    q = np.empty((n, 4))
    tr = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    for k in range(n):
        M = A[k]
        t = tr[k]
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            q[k] = [
                (M[2, 1] - M[1, 2]) / s,
                (M[0, 2] - M[2, 0]) / s,
                (M[1, 0] - M[0, 1]) / s,
                0.25 * s,
            ]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
            q[k] = [
                0.25 * s,
                (M[0, 1] + M[1, 0]) / s,
                (M[0, 2] + M[2, 0]) / s,
                (M[2, 1] - M[1, 2]) / s,
            ]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
            q[k] = [
                (M[0, 1] + M[1, 0]) / s,
                0.25 * s,
                (M[1, 2] + M[2, 1]) / s,
                (M[0, 2] - M[2, 0]) / s,
            ]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
            q[k] = [
                (M[0, 2] + M[2, 0]) / s,
                (M[1, 2] + M[2, 1]) / s,
                0.25 * s,
                (M[1, 0] - M[0, 1]) / s,
            ]
    out = np.asarray(quat_normalize(q)).reshape(-1, 4)
    return _unbatch(out, single)


# --------------------------------------------------------------------------
# SO(3)
# --------------------------------------------------------------------------


def _theta_of(phi: np.ndarray) -> np.ndarray:
    return np.linalg.norm(phi, axis=-1)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation-vector -> unit quaternion."""
    P, single = _as_batch(phi, 3)
    theta = _theta_of(P)
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    # sin(theta/2)/theta
    half = np.where(
        small,
        0.5 - t2 / 48.0 + t2 * t2 / 3840.0 - t2 * t2 * t2 / 645120.0,
        np.sin(0.5 * safe) / safe,
    )
    q = np.empty((P.shape[0], 4))
    q[:, :3] = P * half[:, None]
    q[:, 3] = np.cos(0.5 * theta)
    return _unbatch(np.asarray(quat_normalize(q)).reshape(-1, 4), single)


def so3_log(q: np.ndarray) -> np.ndarray:
    """Unit quaternion -> rotation vector, valid up to ``|phi| = pi``."""
    Q, single = _as_batch(q, 4)
    Q = np.asarray(quat_normalize(Q)).reshape(-1, 4)
    v = Q[:, :3]
    w = np.clip(Q[:, 3], -1.0, 1.0)
    n = np.linalg.norm(v, axis=1)
    small = n < SMALL_NORM
    safe_n = np.where(small, 1.0, n)
    safe_w = np.where(np.abs(w) < 1e-15, 1e-15, w)
    # factor = 2 * atan2(n, w) / n
    big = 2.0 * np.arctan2(n, w) / safe_n
    ser = (2.0 / safe_w) * (1.0 - (n * n) / (3.0 * safe_w * safe_w))
    factor = np.where(small, ser, big)
    return _unbatch(v * factor[:, None], single)


def _coeff_B(theta: np.ndarray) -> np.ndarray:
    """``(1 - cos t)/t^2``."""
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, t2)
    ser = 0.5 - t2 / 24.0 + t2 * t2 / 720.0 - t2 * t2 * t2 / 40320.0
    return np.where(small, ser, (1.0 - np.cos(theta)) / safe)


def _coeff_C(theta: np.ndarray) -> np.ndarray:
    """``(t - sin t)/t^3``."""
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, t2 * theta)
    ser = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362880.0
    return np.where(small, ser, (theta - np.sin(theta)) / safe)


def _coeff_D(theta: np.ndarray) -> np.ndarray:
    """``1/t^2 - cot(t/2)/(2t)``, the quadratic coefficient of ``Jl^-1``.

    Written with ``cot(t/2)`` rather than ``(1+cos t)/(2 t sin t)`` so it stays
    finite at ``t = pi``, which is exactly where near-180-degree loop closures
    live.
    """
    t2 = theta * theta
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    half = 0.5 * safe
    cot = np.cos(half) / np.where(np.abs(np.sin(half)) < 1e-15, 1e-15, np.sin(half))
    big = 1.0 / (safe * safe) - cot / (2.0 * safe)
    ser = (
        1.0 / 12.0
        + t2 / 720.0
        + t2 * t2 / 30240.0
        + t2 * t2 * t2 / 1209600.0
    )
    return np.where(small, ser, big)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): ``I + B(t) [phi]x + C(t) [phi]x^2``."""
    P, single = _as_batch(phi, 3)
    theta = _theta_of(P)
    S = np.asarray(skew(P)).reshape(-1, 3, 3)
    B = _coeff_B(theta)[:, None, None]
    C = _coeff_C(theta)[:, None, None]
    J = np.eye(3)[None] + B * S + C * (S @ S)
    return _unbatch(J, single)


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SO(3): ``I - 0.5 [phi]x + D(t) [phi]x^2``."""
    P, single = _as_batch(phi, 3)
    theta = _theta_of(P)
    S = np.asarray(skew(P)).reshape(-1, 3, 3)
    D = _coeff_D(theta)[:, None, None]
    J = np.eye(3)[None] - 0.5 * S + D * (S @ S)
    return _unbatch(J, single)


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) (``Jr(phi) = Jl(-phi)``)."""
    P, single = _as_batch(phi, 3)
    J = np.asarray(so3_left_jacobian(-P)).reshape(-1, 3, 3)
    return _unbatch(J, single)


# --------------------------------------------------------------------------
# SE(3) group operations
# --------------------------------------------------------------------------


def translation(X: np.ndarray) -> np.ndarray:
    """Translation part of a pose."""
    P, single = _as_batch(X, 7)
    return _unbatch(P[:, :3].copy(), single)


def from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a pose from a rotation matrix and a translation vector."""
    q = np.asarray(rotation_to_quat(R)).reshape(-1, 4)
    T = np.asarray(t, dtype=float).reshape(-1, 3)
    out = np.concatenate([T, q], axis=1)
    return out[0] if (np.ndim(t) == 1 and np.ndim(R) == 2) else out


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Group product ``a * b``."""
    A, sa = _as_batch(a, 7)
    B, sb = _as_batch(b, 7)
    qa, qb = A[:, 3:], B[:, 3:]
    Ra = np.asarray(quat_to_rotation(qa)).reshape(-1, 3, 3)
    t = A[:, :3] + np.einsum("nij,nj->ni", Ra, B[:, :3])
    q = np.asarray(quat_normalize(quat_multiply(qa, qb))).reshape(-1, 4)
    out = np.concatenate([t, q], axis=1)
    return _unbatch(out, sa and sb)


def inverse(a: np.ndarray) -> np.ndarray:
    """Group inverse."""
    A, single = _as_batch(a, 7)
    qi = np.asarray(quat_conjugate(A[:, 3:])).reshape(-1, 4)
    Ri = np.asarray(quat_to_rotation(qi)).reshape(-1, 3, 3)
    t = -np.einsum("nij,nj->ni", Ri, A[:, :3])
    out = np.concatenate([t, quat_normalize(qi).reshape(-1, 4)], axis=1)
    return _unbatch(out, single)


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative pose ``a^-1 * b``."""
    return compose(inverse(a), b)


def exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map ``se(3) -> SE(3)``."""
    X, single = _as_batch(xi, 6)
    rho, phi = X[:, :3], X[:, 3:]
    q = np.asarray(so3_exp(phi)).reshape(-1, 4)
    Jl = np.asarray(so3_left_jacobian(phi)).reshape(-1, 3, 3)
    t = np.einsum("nij,nj->ni", Jl, rho)
    return _unbatch(np.concatenate([t, q], axis=1), single)


def log(X: np.ndarray) -> np.ndarray:
    """Logarithm map ``SE(3) -> se(3)``."""
    P, single = _as_batch(X, 7)
    phi = np.asarray(so3_log(P[:, 3:])).reshape(-1, 3)
    Jli = np.asarray(so3_left_jacobian_inv(phi)).reshape(-1, 3, 3)
    rho = np.einsum("nij,nj->ni", Jli, P[:, :3])
    return _unbatch(np.concatenate([rho, phi], axis=1), single)


def adjoint(X: np.ndarray) -> np.ndarray:
    """Adjoint ``Ad_X`` of shape ``(..., 6, 6)`` for the ``(rho, phi)`` ordering."""
    P, single = _as_batch(X, 7)
    n = P.shape[0]
    R = np.asarray(quat_to_rotation(P[:, 3:])).reshape(-1, 3, 3)
    S = np.asarray(skew(P[:, :3])).reshape(-1, 3, 3)
    Ad = np.zeros((n, 6, 6))
    Ad[:, :3, :3] = R
    Ad[:, :3, 3:] = S @ R
    Ad[:, 3:, 3:] = R
    return _unbatch(Ad, single)


def _little_adjoint(xi: np.ndarray) -> np.ndarray:
    """``ad_xi = [[[phi]x, [rho]x], [0, [phi]x]]`` of shape ``(N, 6, 6)``."""
    n = xi.shape[0]
    ad = np.zeros((n, 6, 6))
    Sp = np.asarray(skew(xi[:, 3:])).reshape(-1, 3, 3)
    Sr = np.asarray(skew(xi[:, :3])).reshape(-1, 3, 3)
    ad[:, :3, :3] = Sp
    ad[:, :3, 3:] = Sr
    ad[:, 3:, 3:] = Sp
    return ad


def _se3_jl_coeffs(theta: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Polynomial coefficients ``(a, b, c, d)`` of the SE(3) left Jacobian."""
    t = theta
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    small = t < SMALL_ANGLE_SE3
    safe = np.where(small, 1.0, t)
    s, c = np.sin(safe), np.cos(safe)
    a_big = (4.0 - safe * s - 4.0 * c) / (2.0 * safe**2)
    b_big = (4.0 * safe - 5.0 * s + safe * c) / (2.0 * safe**3)
    c_big = (2.0 - safe * s - 2.0 * c) / (2.0 * safe**4)
    d_big = (2.0 * safe - 3.0 * s + safe * c) / (2.0 * safe**5)
    a_ser = 0.5 - t4 / 720.0 + t6 / 20160.0
    b_ser = 1.0 / 6.0 - t4 / 5040.0 + t6 / 181440.0
    c_ser = 1.0 / 24.0 - t2 / 360.0 + t4 / 13440.0 - t6 / 907200.0
    d_ser = 1.0 / 120.0 - t2 / 2520.0 + t4 / 120960.0 - t6 / 9979200.0
    return (
        np.where(small, a_ser, a_big),
        np.where(small, b_ser, b_big),
        np.where(small, c_ser, c_big),
        np.where(small, d_ser, d_big),
    )


def left_jacobian(xi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SE(3), shape ``(..., 6, 6)``."""
    X, single = _as_batch(xi, 6)
    ad = _little_adjoint(X)
    theta = _theta_of(X[:, 3:])
    a, b, c, d = (k[:, None, None] for k in _se3_jl_coeffs(theta))
    ad2 = ad @ ad
    ad3 = ad2 @ ad
    ad4 = ad3 @ ad
    J = np.eye(6)[None] + a * ad + b * ad2 + c * ad3 + d * ad4
    return _unbatch(J, single)


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(xi) = Jl(-xi)``."""
    X, single = _as_batch(xi, 6)
    J = np.asarray(left_jacobian(-X)).reshape(-1, 6, 6)
    return _unbatch(J, single)


def left_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SE(3)."""
    X, single = _as_batch(xi, 6)
    J = np.linalg.inv(np.asarray(left_jacobian(X)).reshape(-1, 6, 6))
    return _unbatch(J, single)


def right_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SE(3)."""
    X, single = _as_batch(xi, 6)
    J = np.linalg.inv(np.asarray(right_jacobian(X)).reshape(-1, 6, 6))
    return _unbatch(J, single)


def plus(X: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Box-plus retraction ``X * Exp(delta)``."""
    return compose(X, exp(delta))


def minus(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Box-minus ``Log(Y^-1 X)``."""
    return log(between(Y, X))


def to_matrix(X: np.ndarray) -> np.ndarray:
    """Homogeneous 4x4 matrix form."""
    P, single = _as_batch(X, 7)
    n = P.shape[0]
    M = np.zeros((n, 4, 4))
    M[:, :3, :3] = np.asarray(quat_to_rotation(P[:, 3:])).reshape(-1, 3, 3)
    M[:, :3, 3] = P[:, :3]
    M[:, 3, 3] = 1.0
    return _unbatch(M, single)


def from_matrix(M: np.ndarray) -> np.ndarray:
    """Pose vector from a homogeneous 4x4 matrix."""
    A = np.asarray(M, dtype=float)
    single = A.ndim == 2
    A = A.reshape(-1, 4, 4)
    q = np.asarray(rotation_to_quat(A[:, :3, :3])).reshape(-1, 4)
    out = np.concatenate([A[:, :3, 3], q], axis=1)
    return _unbatch(out, single)


def relative_error(Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Error ``Log(Z^-1 Xi^-1 Xj)`` of a binary relative-pose measurement."""
    return log(between(Z, between(Xi, Xj)))


def relative_error_jacobians(
    Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic error and Jacobians of a binary SE(3) edge.

    Returns ``(e, Ji, Jj)`` with::

        e  = Log(Z^-1 Xi^-1 Xj)
        Ji = -Jr^-1(e) * Ad(Xj^-1 Xi)
        Jj =  Jr^-1(e)
    """
    Pi, si = _as_batch(Xi, 7)
    Pj, sj = _as_batch(Xj, 7)
    Pz, sz = _as_batch(Z, 7)
    e = np.atleast_2d(log(between(Pz, between(Pi, Pj))))
    Jri = np.asarray(right_jacobian_inv(e)).reshape(-1, 6, 6)
    Ad = np.asarray(adjoint(between(Pj, Pi))).reshape(-1, 6, 6)
    single = si and sj and sz
    return _unbatch(e, single), _unbatch(-Jri @ Ad, single), _unbatch(Jri, single)
