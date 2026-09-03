#!/usr/bin/env python3
"""Print the analytic Jacobians next to central differences.

This is the check the whole package rests on, run in the open rather than only
inside the test suite.

    python3 examples/05_check_jacobians.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posegraph import se2, se3  # noqa: E402


def numerical(f, x, plus, dof, eps=1e-6):
    y0 = np.atleast_1d(f(x))
    J = np.zeros((y0.size, dof))
    for k in range(dof):
        dp, dm = np.zeros(dof), np.zeros(dof)
        dp[k], dm[k] = eps, -eps
        J[:, k] = (np.atleast_1d(f(plus(x, dp))) - np.atleast_1d(f(plus(x, dm)))) / (2 * eps)
    return J


def main() -> int:
    rng = np.random.default_rng(0)

    print("SE(2) binary edge, one sample\n")
    Xi, Xj, Z = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    e, Ji, Jj = se2.relative_error_jacobians(Xi, Xj, Z)
    nJi = numerical(lambda x: se2.relative_error(x, Xj, Z), Xi, se2.plus, 3)
    print("error e =", np.array2string(e, precision=6))
    print("\nanalytic dE/dXi:\n", np.array2string(Ji, precision=6, suppress_small=True))
    print("\nnumerical dE/dXi:\n", np.array2string(nJi, precision=6, suppress_small=True))
    print(f"\nmax |difference| = {np.abs(Ji - nJi).max():.3e}")

    print("\n" + "-" * 68)
    print("\nSweeping 500 random configurations of each group\n")
    for label, dof, ops, sample in (
        ("SE(2)", 3, se2, lambda: rng.normal(size=3)),
        ("SE(3)", 6, se3, lambda: se3.exp(rng.normal(size=6))),
    ):
        worst_i = worst_j = 0.0
        for _ in range(500):
            A, B, M = sample(), sample(), sample()
            _e, Ja, Jb = ops.relative_error_jacobians(A, B, M)
            na = numerical(lambda x: ops.relative_error(x, B, M), A, ops.plus, dof)
            nb = numerical(lambda x: ops.relative_error(A, x, M), B, ops.plus, dof)
            worst_i = max(worst_i, float(np.abs(Ja - na).max()))
            worst_j = max(worst_j, float(np.abs(Jb - nb).max()))
        print(f"{label}: worst |analytic - numerical| = {worst_i:.3e} (i), {worst_j:.3e} (j)")
    print(
        "\nCentral differences with a 1e-6 step carry roughly 1e-9 of truncation and\n"
        "round-off themselves, so anything at that level means the analytic form is\n"
        "exact and the difference is the finite-difference error, not ours."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
