"""Nonlinear least-squares back-end: Gauss-Newton, Levenberg-Marquardt, Dogleg.

The problem
-----------
Given poses ``x`` and relative measurements ``z_ij`` with information matrices
``Omega_ij``, minimise

    F(x) = sum_ij rho( e_ij(x)^T Omega_ij e_ij(x) ),
    e_ij(x) = Log( z_ij^-1 * x_i^-1 * x_j )

Linearising ``e`` around the current estimate with the analytic Jacobians from
:mod:`posegraph.se2` / :mod:`posegraph.se3` gives the sparse normal equations

    H dx = b,   H = sum J^T (w Omega) J,   b = -sum J^T (w Omega) e

which are solved for a tangent-space increment and applied with the retraction
``x <- x [+] dx``. ``H`` is singular until the gauge is fixed: a pose graph is
only determined up to a global rigid transform, so at least one pose must be
held fixed or pinned by a prior.

Structure caching
-----------------
The sparsity pattern of ``H`` does not change between iterations, only the
values do. The row/column index arrays, the free-variable map and the
fill-reducing permutation are therefore computed once per :class:`Problem` and
reused, which is most of the reason this runs at a usable speed in Python.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import linalg, se3
from .graph import PoseGraph
from .robust import RobustKernel, default_delta, make_kernel

__all__ = [
    "IterationRecord",
    "SolverOptions",
    "SolverResult",
    "Problem",
    "optimize",
    "gauss_newton",
    "levenberg_marquardt",
    "dogleg",
]


@dataclass
class IterationRecord:
    """One optimiser iteration, kept for the convergence plots."""

    iteration: int
    chi2: float
    cost: float
    step_norm: float
    lambda_or_radius: float
    accepted: bool
    seconds: float


@dataclass
class SolverOptions:
    """Knobs for :func:`optimize`."""

    method: str = "lm"
    max_iterations: int = 100
    #: stop when the relative decrease in the robust cost drops below this
    rel_tolerance: float = 1e-8
    #: stop when the largest component of the increment drops below this
    step_tolerance: float = 1e-9
    #: stop when the gradient infinity norm drops below this
    gradient_tolerance: float = 1e-10
    initial_lambda: float = 1e-4
    max_lambda: float = 1e12
    initial_radius: float = 10.0
    damping: str = "marquardt"
    kernel: Optional[str] = None
    #: Kernel width in units of sqrt(chi2). ``None`` picks
    #: :func:`posegraph.robust.default_delta` for the problem's dimension, which
    #: is the chi-squared 95% critical value and is almost always what you want.
    kernel_delta: Optional[float] = None
    ordering: str = "auto"
    backend: str = "auto"
    verbose: bool = False


@dataclass
class SolverResult:
    """Everything the optimiser measured, so nothing has to be guessed later."""

    converged: bool
    iterations: int
    initial_chi2: float
    final_chi2: float
    initial_cost: float
    final_cost: float
    history: List[IterationRecord] = field(default_factory=list)
    backend: str = ""
    ordering: Optional[linalg.OrderingReport] = None
    seconds: float = 0.0
    message: str = ""

    @property
    def chi2_history(self) -> List[float]:
        return [r.chi2 for r in self.history]

    def summary(self) -> str:
        return (
            f"{'converged' if self.converged else 'stopped'} after {self.iterations} "
            f"iterations: chi2 {self.initial_chi2:.6g} -> {self.final_chi2:.6g} "
            f"in {self.seconds:.2f}s using {self.backend}"
        )


class Problem:
    """A linearisable view of a :class:`~posegraph.graph.PoseGraph`."""

    def __init__(
        self,
        graph: PoseGraph,
        kernel: RobustKernel | str | None = None,
        kernel_delta: Optional[float] = None,
        ordering: str = "auto",
        backend: str = "auto",
    ) -> None:
        self.graph = graph
        if kernel_delta is None:
            kernel_delta = default_delta(graph.dim)
        self.kernel = make_kernel(kernel, kernel_delta)
        self.ops = graph.ops
        self.d = graph.dim
        self.q = graph.point_dim
        self.backend = backend
        self._build_structure(ordering)

    # -- structure ----------------------------------------------------------

    def _build_structure(self, ordering: str) -> None:
        g = self.graph
        d, q = self.d, self.q
        n_pose, n_point = g.num_poses, g.num_points
        self.n_vars = n_pose + n_point
        dims = np.concatenate([np.full(n_pose, d), np.full(n_point, q)]).astype(np.int64)
        self.var_dims = dims
        self.var_offset = np.concatenate([[0], np.cumsum(dims)]).astype(np.int64)
        self.n_scalar = int(self.var_offset[-1])

        fixed_vars = np.zeros(self.n_vars, dtype=bool)
        for nid in g.fixed:
            fixed_vars[g.pose_index(nid)] = True
        self.fixed_vars = fixed_vars
        free_scalar = np.concatenate(
            [
                np.arange(self.var_offset[v], self.var_offset[v + 1])
                for v in range(self.n_vars)
                if not fixed_vars[v]
            ]
        ).astype(np.int64) if (~fixed_vars).any() else np.zeros(0, dtype=np.int64)
        self.free_scalar = free_scalar
        self.n_free = int(free_scalar.size)
        remap = np.full(self.n_scalar, -1, dtype=np.int64)
        remap[free_scalar] = np.arange(self.n_free)
        self.remap = remap

        # free variable indices, and the block graph over them
        free_vars = np.flatnonzero(~fixed_vars)
        var_to_free = np.full(self.n_vars, -1, dtype=np.int64)
        var_to_free[free_vars] = np.arange(free_vars.size)
        self.free_vars = free_vars
        self.n_free_blocks = int(free_vars.size)

        ri, rj = g.edge_rows()
        self.edge_rows_i, self.edge_rows_j = ri, rj
        if g.num_landmark_edges:
            self.lm_rows_p = np.fromiter(
                (g.pose_index(int(i)) for i in g.lm_pose),
                dtype=np.int64,
                count=g.num_landmark_edges,
            )
            self.lm_rows_q = np.fromiter(
                (n_pose + g._point_index[int(i)] for i in g.lm_point),
                dtype=np.int64,
                count=g.num_landmark_edges,
            )
        else:
            self.lm_rows_p = np.zeros(0, dtype=np.int64)
            self.lm_rows_q = np.zeros(0, dtype=np.int64)

        # block adjacency of the free variables, for ordering and fill analysis
        pairs_a: List[np.ndarray] = []
        pairs_b: List[np.ndarray] = []
        if g.num_edges:
            pairs_a.append(var_to_free[ri])
            pairs_b.append(var_to_free[rj])
        if g.num_landmark_edges:
            pairs_a.append(var_to_free[self.lm_rows_p])
            pairs_b.append(var_to_free[self.lm_rows_q])
        if pairs_a:
            aa = np.concatenate(pairs_a)
            bb = np.concatenate(pairs_b)
            keep = (aa >= 0) & (bb >= 0)
            aa, bb = aa[keep], bb[keep]
        else:
            aa = bb = np.zeros(0, dtype=np.int64)
        self.block_adj = linalg.block_adjacency(self.n_free_blocks, aa, bb)

        self.ordering_report: Optional[linalg.OrderingReport] = None
        block_perm = None
        if ordering != "none" and self.n_free_blocks > 1:
            methods = ("natural", "rcm", "md")
            if ordering == "rcm":
                methods = ("natural", "rcm")
            elif ordering == "md":
                methods = ("natural", "md")
            self.ordering_report = linalg.fill_report(self.block_adj, methods)
            block_perm = self.ordering_report.perm

        if block_perm is None:
            self.scalar_perm = None
        else:
            free_dims = dims[free_vars]
            free_off = np.concatenate([[0], np.cumsum(free_dims)]).astype(np.int64)
            self.scalar_perm = np.concatenate(
                [np.arange(free_off[v], free_off[v + 1]) for v in block_perm]
            ).astype(np.int64)

        self._build_index_arrays()

    def _build_index_arrays(self) -> None:
        """Precompute the COO row/column indices of ``H`` (fixed across iterations)."""
        g = self.graph
        d, q = self.d, self.q
        chunks_r: List[np.ndarray] = []
        chunks_c: List[np.ndarray] = []
        ar = np.arange(d)
        if g.num_edges:
            oi = self.var_offset[self.edge_rows_i]
            oj = self.var_offset[self.edge_rows_j]
            for a, b in ((oi, oi), (oi, oj), (oj, oi), (oj, oj)):
                chunks_r.append((a[:, None, None] + ar[None, :, None]).repeat(d, axis=2).ravel())
                chunks_c.append(
                    np.broadcast_to(b[:, None, None] + ar[None, None, :], (b.size, d, d)).ravel()
                )
        aq = np.arange(q)
        if g.num_landmark_edges:
            op = self.var_offset[self.lm_rows_p]
            oq = self.var_offset[self.lm_rows_q]
            # pose-pose (d x d), pose-point (d x q), point-pose (q x d), point-point (q x q)
            chunks_r.append((op[:, None, None] + ar[None, :, None]).repeat(d, axis=2).ravel())
            chunks_c.append(
                np.broadcast_to(op[:, None, None] + ar[None, None, :], (op.size, d, d)).ravel()
            )
            chunks_r.append((op[:, None, None] + ar[None, :, None]).repeat(q, axis=2).ravel())
            chunks_c.append(
                np.broadcast_to(oq[:, None, None] + aq[None, None, :], (oq.size, d, q)).ravel()
            )
            chunks_r.append((oq[:, None, None] + aq[None, :, None]).repeat(d, axis=2).ravel())
            chunks_c.append(
                np.broadcast_to(op[:, None, None] + ar[None, None, :], (op.size, q, d)).ravel()
            )
            chunks_r.append((oq[:, None, None] + aq[None, :, None]).repeat(q, axis=2).ravel())
            chunks_c.append(
                np.broadcast_to(oq[:, None, None] + aq[None, None, :], (oq.size, q, q)).ravel()
            )
        if g.priors:
            pr = np.array([g.pose_index(p.node) for p in g.priors], dtype=np.int64)
            op = self.var_offset[pr]
            chunks_r.append((op[:, None, None] + ar[None, :, None]).repeat(d, axis=2).ravel())
            chunks_c.append(
                np.broadcast_to(op[:, None, None] + ar[None, None, :], (op.size, d, d)).ravel()
            )
            self.prior_rows = pr
        else:
            self.prior_rows = np.zeros(0, dtype=np.int64)

        if chunks_r:
            rows = np.concatenate(chunks_r)
            cols = np.concatenate(chunks_c)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
        rr, cc = self.remap[rows], self.remap[cols]
        keep = (rr >= 0) & (cc >= 0)
        self.H_rows = rr[keep]
        self.H_cols = cc[keep]
        self._keep = keep
        self.diag_mask = self.H_rows == self.H_cols

    # -- linearisation ------------------------------------------------------

    def errors_and_weights(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Binary-edge errors, IRLS weights, chi2 and robust cost."""
        g = self.graph
        e = g.edge_errors()
        if e.size == 0:
            return e, np.zeros(0), 0.0, 0.0
        s = np.einsum("ni,nij,nj->n", e, g.edge_info, e)
        rho, w = self.kernel.evaluate(s)
        return e, w, float(s.sum()), float(rho.sum())

    def total_cost(self) -> Tuple[float, float]:
        """``(chi2, robust cost)`` over binary, landmark and prior edges."""
        g = self.graph
        chi2 = 0.0
        cost = 0.0
        if g.num_edges:
            e = g.edge_errors()
            s = np.einsum("ni,nij,nj->n", e, g.edge_info, e)
            rho, _ = self.kernel.evaluate(s)
            chi2 += float(s.sum())
            cost += float(rho.sum())
        if g.num_landmark_edges:
            le = g.landmark_errors()
            s = np.einsum("ni,nij,nj->n", le, g.lm_info, le)
            rho, _ = self.kernel.evaluate(s)
            chi2 += float(s.sum())
            cost += float(rho.sum())
        for pe, p in zip(g.prior_errors(), g.priors):
            v = float(pe @ p.information @ pe)
            chi2 += v
            cost += v
        return chi2, cost

    def build_system(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Assemble ``(data, b, chi2, cost)`` for the current estimate."""
        g = self.graph
        d, q = self.d, self.q
        ops = self.ops
        data_chunks: List[np.ndarray] = []
        b_full = np.zeros(self.n_scalar)
        chi2 = 0.0
        cost = 0.0

        if g.num_edges:
            Xi = g.poses[self.edge_rows_i]
            Xj = g.poses[self.edge_rows_j]
            e, Ji, Jj = ops.relative_error_jacobians(Xi, Xj, g.edge_z)
            e = np.atleast_2d(e)
            omega = g.edge_info
            s = np.einsum("ni,nij,nj->n", e, omega, e)
            rho, w = self.kernel.evaluate(s)
            chi2 += float(s.sum())
            cost += float(rho.sum())
            W = w[:, None, None] * omega
            JiTW = np.einsum("nki,nkl->nil", Ji, W)
            JjTW = np.einsum("nki,nkl->nil", Jj, W)
            Hii = JiTW @ Ji
            Hij = JiTW @ Jj
            Hjj = JjTW @ Jj
            data_chunks += [Hii.ravel(), Hij.ravel(), Hij.transpose(0, 2, 1).ravel(), Hjj.ravel()]
            gi = -np.einsum("nil,nl->ni", JiTW, e)
            gj = -np.einsum("nil,nl->ni", JjTW, e)
            ar = np.arange(d)
            idx_i = (self.var_offset[self.edge_rows_i][:, None] + ar[None, :]).ravel()
            idx_j = (self.var_offset[self.edge_rows_j][:, None] + ar[None, :]).ravel()
            b_full += np.bincount(idx_i, weights=gi.ravel(), minlength=self.n_scalar)
            b_full += np.bincount(idx_j, weights=gj.ravel(), minlength=self.n_scalar)

        if g.num_landmark_edges:
            le = g.landmark_errors()
            omega = g.lm_info
            s = np.einsum("ni,nij,nj->n", le, omega, le)
            rho, w = self.kernel.evaluate(s)
            chi2 += float(s.sum())
            cost += float(rho.sum())
            Jp, Jq = self._landmark_jacobians()
            W = w[:, None, None] * omega
            JpTW = np.einsum("nki,nkl->nil", Jp, W)
            JqTW = np.einsum("nki,nkl->nil", Jq, W)
            Hpp = JpTW @ Jp
            Hpq = JpTW @ Jq
            Hqq = JqTW @ Jq
            data_chunks += [
                Hpp.ravel(),
                Hpq.ravel(),
                Hpq.transpose(0, 2, 1).ravel(),
                Hqq.ravel(),
            ]
            gp = -np.einsum("nil,nl->ni", JpTW, le)
            gq = -np.einsum("nil,nl->ni", JqTW, le)
            ar, aq = np.arange(d), np.arange(q)
            b_full += np.bincount(
                (self.var_offset[self.lm_rows_p][:, None] + ar[None, :]).ravel(),
                weights=gp.ravel(),
                minlength=self.n_scalar,
            )
            b_full += np.bincount(
                (self.var_offset[self.lm_rows_q][:, None] + aq[None, :]).ravel(),
                weights=gq.ravel(),
                minlength=self.n_scalar,
            )

        if g.priors:
            Hs = []
            for k, p in enumerate(g.priors):
                row = self.prior_rows[k]
                X = g.poses[row]
                err = np.atleast_1d(ops.log(ops.between(p.measurement, X)))
                J = np.asarray(ops.right_jacobian_inv(err)).reshape(d, d)
                JTW = J.T @ p.information
                Hs.append(JTW @ J)
                v = float(err @ p.information @ err)
                chi2 += v
                cost += v
                start = int(self.var_offset[row])
                b_full[start:start + d] += -(JTW @ err)
            data_chunks.append(np.asarray(Hs).ravel())

        data = np.concatenate(data_chunks) if data_chunks else np.zeros(0)
        return data[self._keep], b_full[self.free_scalar], chi2, cost

    def _landmark_jacobians(self) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic Jacobians of the pose-to-point measurement."""
        g = self.graph
        X = g.poses[self.lm_rows_p]
        P = g.points[self.lm_rows_q - g.num_poses]
        m = X.shape[0]
        if self.space_is_se2:
            c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
            R = np.empty((m, 2, 2))
            R[:, 0, 0], R[:, 0, 1] = c, s
            R[:, 1, 0], R[:, 1, 1] = -s, c
            dx = P[:, 0] - X[:, 0]
            dy = P[:, 1] - X[:, 1]
            qv = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
            Jp = np.zeros((m, 2, 3))
            Jp[:, 0, 0] = -1.0
            Jp[:, 1, 1] = -1.0
            Jp[:, 0, 2] = qv[:, 1]
            Jp[:, 1, 2] = -qv[:, 0]
            return Jp, R
        R = np.asarray(se3.quat_to_rotation(X[:, 3:])).reshape(-1, 3, 3)
        qv = np.einsum("nji,nj->ni", R, P - X[:, :3])
        Jp = np.zeros((m, 3, 6))
        Jp[:, :, :3] = -np.eye(3)[None]
        Jp[:, :, 3:] = np.asarray(se3.skew(qv)).reshape(-1, 3, 3)
        return Jp, R.transpose(0, 2, 1)

    @property
    def space_is_se2(self) -> bool:
        return self.graph.space == "SE2"

    # -- stepping -----------------------------------------------------------

    def apply_step(self, dx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retract ``dx`` onto a copy of the current estimate."""
        g = self.graph
        d, q = self.d, self.q
        full = np.zeros(self.n_scalar)
        full[self.free_scalar] = dx
        poses = g.poses.copy()
        if g.num_poses:
            # every pose variable has the same width, so the pose block of the
            # increment vector reshapes directly
            deltas = full[: g.num_poses * d].reshape(g.num_poses, d)
            poses = np.atleast_2d(self.ops.plus(poses, deltas))
        points = g.points.copy()
        if g.num_points:
            start = g.num_poses * d
            width = g.num_points * q
            points = points + full[start:start + width].reshape(g.num_points, q)
        return poses, points


# --------------------------------------------------------------------------
# optimisers
# --------------------------------------------------------------------------


def _solve_damped(
    problem: Problem,
    data: np.ndarray,
    b: np.ndarray,
    lam: float,
    damping: str,
) -> Tuple[np.ndarray, str, np.ndarray]:
    n = problem.n_free
    diag_idx = problem.diag_mask
    diag = np.bincount(problem.H_rows[diag_idx], weights=data[diag_idx], minlength=n)
    if damping == "levenberg":
        d_add = np.full(n, lam)
    else:
        d_add = lam * np.maximum(diag, 1e-12)
    rows = np.concatenate([problem.H_rows, np.arange(n)])
    cols = np.concatenate([problem.H_cols, np.arange(n)])
    vals = np.concatenate([data, d_add])
    dx, backend = linalg.solve_spd(
        rows, cols, vals, n, b, perm=problem.scalar_perm, backend=problem.backend
    )
    return dx, backend, d_add


def _hessian_times(problem: Problem, data: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``H @ v`` straight from the COO triplets, without materialising ``H``."""
    return np.bincount(problem.H_rows, weights=data * v[problem.H_cols], minlength=v.size)


def optimize(
    graph: PoseGraph,
    options: SolverOptions | None = None,
    callback: Optional[Callable[[IterationRecord], None]] = None,
    **kwargs,
) -> SolverResult:
    """Optimise ``graph`` in place. Returns a :class:`SolverResult`.

    ``kwargs`` are forwarded to :class:`SolverOptions`, so
    ``optimize(g, method="lm", kernel="huber")`` works.
    """
    opts = options or SolverOptions(**kwargs)
    method = opts.method.lower()
    if method in ("gn", "gauss_newton"):
        return gauss_newton(graph, opts, callback)
    if method in ("lm", "levenberg_marquardt"):
        return levenberg_marquardt(graph, opts, callback)
    if method == "dogleg":
        return dogleg(graph, opts, callback)
    raise ValueError(f"unknown method {opts.method!r}")


def _prepare(graph: PoseGraph, opts: SolverOptions) -> Problem:
    if not graph.fixed and not graph.priors:
        raise ValueError(
            "the graph has no fixed pose and no prior: H is singular because the "
            "solution is only defined up to a global rigid transform. "
            "Call graph.fix_pose(id) or graph.add_prior(...)."
        )
    return Problem(
        graph,
        kernel=opts.kernel,
        kernel_delta=opts.kernel_delta,
        ordering=opts.ordering,
        backend=opts.backend,
    )


def gauss_newton(
    graph: PoseGraph,
    opts: SolverOptions | None = None,
    callback: Optional[Callable[[IterationRecord], None]] = None,
) -> SolverResult:
    """Plain Gauss-Newton. Fast when the initial guess is good, divergent when it is not."""
    opts = opts or SolverOptions(method="gn")
    problem = _prepare(graph, opts)
    t0 = time.perf_counter()
    chi2_0, cost_0 = problem.total_cost()
    history: List[IterationRecord] = []
    backend = ""
    cost_prev = cost_0
    converged = False
    message = "iteration limit reached"
    it = 0
    for it in range(1, opts.max_iterations + 1):
        ti = time.perf_counter()
        data, b, chi2, cost = problem.build_system()
        if np.max(np.abs(b)) < opts.gradient_tolerance:
            converged, message = True, "gradient below tolerance"
            break
        dx, backend, _ = _solve_damped(problem, data, b, 0.0, "levenberg")
        poses, points = problem.apply_step(dx)
        graph.poses, graph.points = poses, points
        chi2_new, cost_new = problem.total_cost()
        step = float(np.max(np.abs(dx))) if dx.size else 0.0
        rec = IterationRecord(it, chi2_new, cost_new, step, 0.0, True, time.perf_counter() - ti)
        history.append(rec)
        if callback:
            callback(rec)
        if opts.verbose:
            print(f"  gn it={it:3d} chi2={chi2_new:.6g} |dx|inf={step:.3e}")
        if step < opts.step_tolerance:
            converged, message = True, "step below tolerance"
            break
        if cost_prev > 0 and abs(cost_prev - cost_new) / max(cost_prev, 1e-30) < opts.rel_tolerance:
            converged, message = True, "relative cost change below tolerance"
            cost_prev = cost_new
            break
        cost_prev = cost_new
    chi2_f, cost_f = problem.total_cost()
    return SolverResult(
        converged=converged,
        iterations=len(history),
        initial_chi2=chi2_0,
        final_chi2=chi2_f,
        initial_cost=cost_0,
        final_cost=cost_f,
        history=history,
        backend=backend,
        ordering=problem.ordering_report,
        seconds=time.perf_counter() - t0,
        message=message,
    )


def levenberg_marquardt(
    graph: PoseGraph,
    opts: SolverOptions | None = None,
    callback: Optional[Callable[[IterationRecord], None]] = None,
) -> SolverResult:
    """Levenberg-Marquardt with the Nielsen gain-ratio damping policy.

    ``lambda`` is scaled by ``max(1/3, 1 - (2 rho - 1)^3)`` on an accepted step
    and multiplied by a doubling factor ``nu`` on a rejected one, where ``rho``
    is the ratio of actual to predicted cost reduction. Rejected steps are
    undone exactly, so the reported cost sequence is monotonically
    non-increasing by construction.
    """
    opts = opts or SolverOptions(method="lm")
    problem = _prepare(graph, opts)
    t0 = time.perf_counter()
    chi2_0, cost_0 = problem.total_cost()
    history: List[IterationRecord] = []
    lam = opts.initial_lambda
    nu = 2.0
    backend = ""
    converged = False
    message = "iteration limit reached"
    cost_cur = cost_0

    for it in range(1, opts.max_iterations + 1):
        ti = time.perf_counter()
        data, b, chi2, cost_cur = problem.build_system()
        if np.max(np.abs(b)) < opts.gradient_tolerance:
            converged, message = True, "gradient below tolerance"
            break
        accepted = False
        step = 0.0
        chi2_new, cost_new = chi2, cost_cur
        for _inner in range(12):
            try:
                dx, backend, d_add = _solve_damped(problem, data, b, lam, opts.damping)
            except np.linalg.LinAlgError:
                lam = min(lam * nu, opts.max_lambda)
                nu *= 2.0
                continue
            if not np.all(np.isfinite(dx)):
                lam = min(lam * nu, opts.max_lambda)
                nu *= 2.0
                continue
            predicted = float(dx @ (b + d_add * dx))
            saved_poses, saved_points = graph.poses, graph.points
            poses, points = problem.apply_step(dx)
            graph.poses, graph.points = poses, points
            trial_chi2, trial_cost = problem.total_cost()
            actual = cost_cur - trial_cost
            gain = actual / predicted if predicted > 0 else -1.0
            if gain > 0 and actual > 0:
                accepted = True
                chi2_new, cost_new = trial_chi2, trial_cost
                lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3), 1e-15)
                nu = 2.0
                step = float(np.max(np.abs(dx)))
                break
            graph.poses, graph.points = saved_poses, saved_points
            lam = min(lam * nu, opts.max_lambda)
            nu *= 2.0
            if lam >= opts.max_lambda:
                break
        rec = IterationRecord(
            it, chi2_new, cost_new, step, lam, accepted, time.perf_counter() - ti
        )
        history.append(rec)
        if callback:
            callback(rec)
        if opts.verbose:
            print(
                f"  lm it={it:3d} chi2={chi2_new:.6g} lambda={lam:.3e} "
                f"|dx|inf={step:.3e} {'ok' if accepted else 'rejected'}"
            )
        if not accepted:
            converged, message = False, "no acceptable step (lambda hit its ceiling)"
            break
        if step < opts.step_tolerance:
            converged, message = True, "step below tolerance"
            break
        rel = abs(cost_cur - cost_new) / max(cost_cur, 1e-30)
        if rel < opts.rel_tolerance:
            converged, message = True, "relative cost change below tolerance"
            break

    chi2_f, cost_f = problem.total_cost()
    return SolverResult(
        converged=converged,
        iterations=len(history),
        initial_chi2=chi2_0,
        final_chi2=chi2_f,
        initial_cost=cost_0,
        final_cost=cost_f,
        history=history,
        backend=backend,
        ordering=problem.ordering_report,
        seconds=time.perf_counter() - t0,
        message=message,
    )


def dogleg(
    graph: PoseGraph,
    opts: SolverOptions | None = None,
    callback: Optional[Callable[[IterationRecord], None]] = None,
) -> SolverResult:
    """Powell's dogleg with an explicit trust region.

    Blends the Gauss-Newton step with the Cauchy (steepest-descent) step so the
    step length is controlled by a radius instead of a damping term. One
    factorisation per iteration, no inner retry loop, which makes it cheaper
    than LM per iteration when steps are usually accepted.
    """
    opts = opts or SolverOptions(method="dogleg")
    problem = _prepare(graph, opts)
    t0 = time.perf_counter()
    chi2_0, cost_0 = problem.total_cost()
    history: List[IterationRecord] = []
    radius = opts.initial_radius
    backend = ""
    converged = False
    message = "iteration limit reached"

    for it in range(1, opts.max_iterations + 1):
        ti = time.perf_counter()
        data, b, chi2, cost_cur = problem.build_system()
        gnorm = float(np.max(np.abs(b))) if b.size else 0.0
        if gnorm < opts.gradient_tolerance:
            converged, message = True, "gradient below tolerance"
            break
        try:
            h_gn, backend, _ = _solve_damped(problem, data, b, 1e-12, "levenberg")
        except np.linalg.LinAlgError:
            converged, message = False, "factorisation failed"
            break
        Hb = _hessian_times(problem, data, b)
        denom = float(b @ Hb)
        alpha = float(b @ b) / denom if denom > 0 else 0.0
        h_sd = alpha * b

        n_gn = float(np.linalg.norm(h_gn))
        n_sd = float(np.linalg.norm(h_sd))
        if n_gn <= radius:
            dx = h_gn
        elif n_sd >= radius:
            dx = h_sd * (radius / max(n_sd, 1e-30))
        else:
            diff = h_gn - h_sd
            a = float(diff @ diff)
            bb = 2.0 * float(h_sd @ diff)
            cc = n_sd * n_sd - radius * radius
            disc = max(bb * bb - 4.0 * a * cc, 0.0)
            beta = (-bb + np.sqrt(disc)) / (2.0 * a) if a > 0 else 0.0
            dx = h_sd + beta * diff

        predicted = 2.0 * float(dx @ b) - float(dx @ _hessian_times(problem, data, dx))
        saved_poses, saved_points = graph.poses, graph.points
        poses, points = problem.apply_step(dx)
        graph.poses, graph.points = poses, points
        _c2, cost_new = problem.total_cost()
        actual = cost_cur - cost_new
        gain = actual / predicted if predicted > 0 else -1.0
        accepted = gain > 0 and actual > 0
        if not accepted:
            graph.poses, graph.points = saved_poses, saved_points
            radius *= 0.5
        elif gain > 0.75:
            radius = max(radius, 3.0 * float(np.linalg.norm(dx)))
        elif gain < 0.25:
            radius *= 0.5

        chi2_new, cost_after = problem.total_cost()
        step = float(np.max(np.abs(dx))) if accepted else 0.0
        rec = IterationRecord(
            it, chi2_new, cost_after, step, radius, accepted, time.perf_counter() - ti
        )
        history.append(rec)
        if callback:
            callback(rec)
        if opts.verbose:
            print(
                f"  dl it={it:3d} chi2={chi2_new:.6g} radius={radius:.3e} "
                f"{'ok' if accepted else 'rejected'}"
            )
        if radius < 1e-14:
            converged, message = True, "trust region collapsed"
            break
        if accepted and step < opts.step_tolerance:
            converged, message = True, "step below tolerance"
            break
        if accepted and abs(actual) / max(cost_cur, 1e-30) < opts.rel_tolerance:
            converged, message = True, "relative cost change below tolerance"
            break

    chi2_f, cost_f = problem.total_cost()
    return SolverResult(
        converged=converged,
        iterations=len(history),
        initial_chi2=chi2_0,
        final_chi2=chi2_f,
        initial_cost=cost_0,
        final_cost=cost_f,
        history=history,
        backend=backend,
        ordering=problem.ordering_report,
        seconds=time.perf_counter() - t0,
        message=message,
    )
