"""Figures. Matplotlib is imported lazily so the package still works without it.

Nothing else in :mod:`posegraph` imports this module, and this module imports
matplotlib only inside the functions that need it. Installing the package
without matplotlib is therefore fine; only the plotting calls will fail, and
they fail with a clear message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "plot_trajectory",
    "plot_before_after",
    "plot_convergence",
    "plot_kernel_comparison",
    "plot_sparsity",
    "plot_fill_pattern",
    "plot_residual_histogram",
]

_LINE = 0.6
_DPI = 110


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install matplotlib"
        ) from exc
    return plt


def _xy(poses: np.ndarray, space: str) -> Tuple[np.ndarray, np.ndarray]:
    P = np.atleast_2d(np.asarray(poses, dtype=float))
    return (P[:, 0], P[:, 1])


def _draw(ax, poses: np.ndarray, space: str, colour: str, label: str, lw: float = _LINE) -> None:
    x, y = _xy(poses, space)
    ax.plot(x, y, "-", color=colour, linewidth=lw, label=label, solid_joinstyle="round")


def _finish(ax, title: str, space: str = "SE2") -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _projection_note(space: str) -> str:
    return " (x-y projection of a 3D trajectory)" if space == "SE3" else ""


def plot_trajectory(
    poses: np.ndarray,
    space: str,
    path: str | Path,
    title: str = "trajectory",
    ground_truth: Optional[np.ndarray] = None,
) -> Path:
    """Single trajectory, optionally with ground truth underneath."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    if ground_truth is not None:
        _draw(ax, ground_truth, space, "#9aa4b1", "ground truth", lw=1.0)
    _draw(ax, poses, space, "#1f77b4", "estimate")
    ax.legend(fontsize=7, frameon=False)
    _finish(ax, title + _projection_note(space), space)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_before_after(
    before: np.ndarray,
    after: np.ndarray,
    space: str,
    path: str | Path,
    title: str = "",
    ground_truth: Optional[np.ndarray] = None,
    labels: Tuple[str, str] = ("initial guess", "optimised"),
) -> Path:
    """The classic side-by-side: tangled odometry on the left, clean map on the right."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
    for ax, poses, label, colour in (
        (axes[0], before, labels[0], "#c44e52"),
        (axes[1], after, labels[1], "#1f77b4"),
    ):
        if ground_truth is not None:
            _draw(ax, ground_truth, space, "#c8ced6", "ground truth", lw=1.0)
        _draw(ax, poses, space, colour, label)
        ax.legend(fontsize=7, frameon=False, loc="best")
        _finish(ax, label, space)
    if title:
        fig.suptitle(title + _projection_note(space), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_convergence(
    histories: Dict[str, Sequence[float]],
    path: str | Path,
    title: str = "chi-squared convergence",
    initial: Optional[Dict[str, float]] = None,
) -> Path:
    """chi-squared against iteration on a log axis, one line per solver run."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, hist in histories.items():
        ys = list(hist)
        xs = list(range(1, len(ys) + 1))
        if initial and name in initial:
            ys = [initial[name]] + ys
            xs = [0] + xs
        ax.semilogy(xs, ys, marker="o", markersize=2.5, linewidth=1.0, label=name)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("chi-squared", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_kernel_comparison(
    panels: Sequence[Tuple[str, np.ndarray]],
    space: str,
    path: str | Path,
    title: str = "",
    ground_truth: Optional[np.ndarray] = None,
) -> Path:
    """One panel per kernel, same graph, same injected outliers."""
    plt = _mpl()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (label, poses) in zip(axes, panels):
        if ground_truth is not None:
            _draw(ax, ground_truth, space, "#c8ced6", "ground truth", lw=1.0)
        _draw(ax, poses, space, "#1f77b4", label)
        ax.legend(fontsize=7, frameon=False, loc="best")
        _finish(ax, label, space)
    if title:
        fig.suptitle(title + _projection_note(space), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_sparsity(
    adjacency: List[set],
    perm: Optional[np.ndarray],
    path: str | Path,
    title: str = "information matrix block pattern",
    labels: Tuple[str, str] = ("natural order", "reordered"),
    fill: Optional[Tuple[int, int]] = None,
) -> Path:
    """Block sparsity of ``H`` before and after the fill-reducing permutation."""
    plt = _mpl()
    n = len(adjacency)
    rows: List[int] = []
    cols: List[int] = []
    for u, nbrs in enumerate(adjacency):
        rows.append(u)
        cols.append(u)
        for v in nbrs:
            rows.append(u)
            cols.append(v)
    r = np.asarray(rows)
    c = np.asarray(cols)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
    marker = max(0.15, min(2.0, 400.0 / max(n, 1)))
    axes[0].plot(c, r, "s", markersize=marker, color="#33415c", markeredgewidth=0)
    sub0 = labels[0] if fill is None else f"{labels[0]}\nL blocks = {fill[0]:,}"
    axes[0].set_title(sub0, fontsize=9)
    if perm is not None:
        pos = np.empty(n, dtype=np.int64)
        pos[np.asarray(perm)] = np.arange(n)
        axes[1].plot(pos[c], pos[r], "s", markersize=marker, color="#33415c", markeredgewidth=0)
        sub1 = labels[1] if fill is None else f"{labels[1]}\nL blocks = {fill[1]:,}"
        axes[1].set_title(sub1, fontsize=9)
    for ax in axes:
        ax.set_xlim(-1, n)
        ax.set_ylim(n, -1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_fill_pattern(
    adjacency: List[set],
    perm: np.ndarray,
    path: str | Path,
    title: str = "Cholesky factor, before and after reordering",
    fill: Optional[Tuple[int, int]] = None,
) -> Path:
    """Nonzero pattern of the Cholesky factor in two orderings.

    This is the picture the fill-in numbers describe: every dot is a block that
    the factorisation has to store and touch. The reordered panel is the same
    problem, solved in a different sequence.
    """
    from . import linalg

    plt = _mpl()
    n = len(adjacency)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    marker = max(0.12, min(1.6, 320.0 / max(n, 1)))
    for ax, p, label in (
        (axes[0], np.arange(n), "natural order"),
        (axes[1], np.asarray(perm), "fill-reducing order"),
    ):
        r, c, complete = linalg.symbolic_pattern(adjacency, p)
        ax.plot(c, r, "s", markersize=marker, color="#33415c", markeredgewidth=0)
        count = f"{len(r):,}" + ("" if complete else "+ (truncated)")
        ax.set_title(f"{label}\nnonzero blocks in L = {count}", fontsize=9)
        ax.set_xlim(-1, n)
        ax.set_ylim(n, -1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)


def plot_residual_histogram(
    chi2_per_edge: np.ndarray,
    path: str | Path,
    title: str = "per-edge residuals",
    highlight: Optional[np.ndarray] = None,
) -> Path:
    """Log-scale histogram of per-edge chi-squared, with outliers marked."""
    plt = _mpl()
    v = np.asarray(chi2_per_edge, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 1e-12)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 60)
    ax.hist(v, bins=bins, color="#4c72b0", alpha=0.85, label="all edges")
    if highlight is not None and len(highlight):
        hv = np.maximum(np.asarray(chi2_per_edge, dtype=float)[highlight], 1e-12)
        ax.hist(hv, bins=bins, color="#c44e52", alpha=0.9, label="injected outliers")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edge chi-squared", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return Path(path)
