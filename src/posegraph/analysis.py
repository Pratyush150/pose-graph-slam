"""Diagnostics: residuals, outlier ranking, and trajectory error against truth.

Two different questions live here.

*Did the optimiser do its job?* -- chi-squared totals, per-edge residuals and
a ranked list of the edges that still disagree with the solution. On a healthy
graph the total chi-squared should land near the number of degrees of freedom;
a handful of edges carrying most of the cost is the signature of a bad loop
closure rather than of bad tuning.

*Is the map actually right?* -- that needs ground truth, and the standard
measures are Absolute Trajectory Error (global consistency, after aligning the
two trajectories with Umeyama) and Relative Pose Error (local consistency,
independent of any alignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import se2, se3
from .graph import PoseGraph

__all__ = [
    "ChiSquaredReport",
    "EdgeResidual",
    "TrajectoryError",
    "chi2_report",
    "edge_residuals",
    "rank_outliers",
    "umeyama",
    "positions",
    "absolute_trajectory_error",
    "relative_pose_error",
]


@dataclass
class ChiSquaredReport:
    """Total cost split by edge class, with the degrees-of-freedom yardstick."""

    total: float
    odometry: float
    loop: float
    landmark: float
    prior: float
    num_edges: int
    dof: int

    @property
    def normalised(self) -> float:
        """chi2 per degree of freedom. Near 1.0 means the noise model fits."""
        return self.total / self.dof if self.dof > 0 else float("nan")

    def describe(self) -> str:
        return (
            f"chi2 total={self.total:.6g} (odometry={self.odometry:.6g}, "
            f"loop={self.loop:.6g}, landmark={self.landmark:.6g}, prior={self.prior:.6g}) "
            f"dof={self.dof} chi2/dof={self.normalised:.4g}"
        )


@dataclass
class EdgeResidual:
    """One edge's contribution to the cost."""

    index: int
    i: int
    j: int
    tag: str
    chi2: float
    error: np.ndarray

    @property
    def mahalanobis(self) -> float:
        """``sqrt(chi2)`` -- the residual in standard deviations."""
        return float(np.sqrt(max(self.chi2, 0.0)))


def chi2_report(graph: PoseGraph) -> ChiSquaredReport:
    """Split the total chi-squared by edge class."""
    per_edge = graph.edge_chi2()
    tags = np.asarray(graph.edge_tag) if graph.edge_tag else np.zeros(0, dtype=str)
    odo = float(per_edge[tags == "odometry"].sum()) if per_edge.size else 0.0
    loop = float(per_edge.sum() - odo) if per_edge.size else 0.0
    lm = 0.0
    le = graph.landmark_errors()
    if le.size:
        lm = float(np.einsum("ni,nij,nj->", le, graph.lm_info, le))
    prior = 0.0
    for pe, p in zip(graph.prior_errors(), graph.priors):
        prior += float(pe @ p.information @ pe)
    d = graph.dim
    q = graph.point_dim
    measurements = graph.num_edges * d + graph.num_landmark_edges * q + len(graph.priors) * d
    unknowns = (graph.num_poses - len(graph.fixed)) * d + graph.num_points * q
    dof = max(measurements - unknowns, 1)
    return ChiSquaredReport(
        total=float(per_edge.sum()) + lm + prior,
        odometry=odo,
        loop=loop,
        landmark=lm,
        prior=prior,
        num_edges=graph.num_edges,
        dof=dof,
    )


def edge_residuals(graph: PoseGraph) -> List[EdgeResidual]:
    """Per-edge residuals in graph order."""
    err = graph.edge_errors()
    chi = graph.edge_chi2()
    out = []
    for k in range(graph.num_edges):
        out.append(
            EdgeResidual(
                index=k,
                i=int(graph.edge_i[k]),
                j=int(graph.edge_j[k]),
                tag=graph.edge_tag[k] if k < len(graph.edge_tag) else "",
                chi2=float(chi[k]),
                error=err[k].copy(),
            )
        )
    return out


def rank_outliers(graph: PoseGraph, top: int = 20) -> List[EdgeResidual]:
    """The ``top`` edges with the largest residual, worst first.

    This is the list to look at when a map comes out warped: a real outlier
    usually sits orders of magnitude above the median, not just a bit above it.
    """
    res = edge_residuals(graph)
    res.sort(key=lambda r: r.chi2, reverse=True)
    return res[: max(top, 0)]


# --------------------------------------------------------------------------
# trajectory error
# --------------------------------------------------------------------------


def positions(poses: np.ndarray, space: str) -> np.ndarray:
    """Translation columns of a pose array."""
    P = np.atleast_2d(np.asarray(poses, dtype=float))
    return P[:, :2].copy() if space == "SE2" else P[:, :3].copy()


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Implements Umeyama (1991), including the reflection guard: if the
    determinant of the correlation matrix is negative the smallest singular
    direction is flipped, so the result is always a rotation and never a
    mirror image. Returns ``(R, t, s)`` with ``dst ~ s * R @ src + t``.
    """
    S = np.atleast_2d(np.asarray(src, dtype=float))
    D = np.atleast_2d(np.asarray(dst, dtype=float))
    if S.shape != D.shape:
        raise ValueError("point sets must have the same shape")
    n, dim = S.shape
    if n == 0:
        raise ValueError("need at least one point")
    mu_s, mu_d = S.mean(axis=0), D.mean(axis=0)
    Sc, Dc = S - mu_s, D - mu_d
    cov = (Dc.T @ Sc) / n
    U, sing, Vt = np.linalg.svd(cov)
    W = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[-1, -1] = -1.0
    R = U @ W @ Vt
    if with_scale:
        var_s = float((Sc**2).sum()) / n
        scale = float(np.trace(np.diag(sing) @ W) / var_s) if var_s > 0 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return R, t, scale


@dataclass
class TrajectoryError:
    """Summary statistics of a trajectory comparison."""

    rmse: float
    mean: float
    median: float
    std: float
    max: float
    n: int
    rotation_rmse_deg: Optional[float] = None
    label: str = "ate"

    def describe(self) -> str:
        rot = (
            f", rotation rmse={self.rotation_rmse_deg:.4f} deg"
            if self.rotation_rmse_deg is not None
            else ""
        )
        return (
            f"{self.label}: rmse={self.rmse:.4f} mean={self.mean:.4f} "
            f"median={self.median:.4f} max={self.max:.4f} over {self.n} poses{rot}"
        )


def _stats(errors: np.ndarray, label: str, rot_deg: Optional[np.ndarray] = None) -> TrajectoryError:
    e = np.asarray(errors, dtype=float)
    return TrajectoryError(
        rmse=float(np.sqrt(np.mean(e**2))),
        mean=float(np.mean(e)),
        median=float(np.median(e)),
        std=float(np.std(e)),
        max=float(np.max(e)),
        n=int(e.size),
        rotation_rmse_deg=(
            float(np.degrees(np.sqrt(np.mean(np.asarray(rot_deg, dtype=float) ** 2))))
            if rot_deg is not None
            else None
        ),
        label=label,
    )


def absolute_trajectory_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    space: str,
    align: bool = True,
    with_scale: bool = False,
) -> TrajectoryError:
    """Absolute Trajectory Error after optional Umeyama alignment.

    A pose graph is only determined up to a global rigid transform, so the
    estimate must be aligned to the ground truth before the two are compared.
    Skipping the alignment (``align=False``) measures something else entirely:
    how far the estimate has drifted from the ground-truth *frame*.
    """
    P = positions(estimated, space)
    G = positions(ground_truth, space)
    if P.shape != G.shape:
        raise ValueError(f"trajectory shapes differ: {P.shape} vs {G.shape}")
    if align:
        R, t, s = umeyama(P, G, with_scale=with_scale)
        P = (s * (R @ P.T)).T + t
    err = np.linalg.norm(P - G, axis=1)
    return _stats(err, "ate")


def relative_pose_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    space: str,
    delta: int = 1,
) -> TrajectoryError:
    """Relative Pose Error over a fixed pose-index gap.

    Compares ``est_k^-1 est_{k+delta}`` with the same quantity from ground
    truth. No alignment is involved, so this measures local consistency (drift
    rate) rather than global consistency.
    """
    ops = se2 if space == "SE2" else se3
    E = np.atleast_2d(np.asarray(estimated, dtype=float))
    G = np.atleast_2d(np.asarray(ground_truth, dtype=float))
    if E.shape != G.shape:
        raise ValueError("trajectory shapes differ")
    if delta < 1 or E.shape[0] <= delta:
        raise ValueError("delta must be >= 1 and shorter than the trajectory")
    est_rel = ops.between(E[:-delta], E[delta:])
    gt_rel = ops.between(G[:-delta], G[delta:])
    # the residual pose itself, so the translation part is a real distance
    # rather than the Jacobian-weighted translation the logarithm returns
    res = np.atleast_2d(ops.between(gt_rel, est_rel))
    if space == "SE2":
        trans = np.linalg.norm(res[:, :2], axis=1)
        rot = np.abs(se2.normalize_angle(res[:, 2]))
    else:
        trans = np.linalg.norm(res[:, :3], axis=1)
        rot = np.linalg.norm(np.atleast_2d(se3.so3_log(res[:, 3:])), axis=1)
    return _stats(trans, f"rpe(delta={delta})", rot)
