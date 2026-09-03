"""Robust kernels for pose-graph optimisation.

Why this file exists: one wrong loop closure is enough to destroy an entire
map. A plain least-squares back-end will happily fold a 30-metre lie into the
solution because a squared cost grows without bound and the optimiser can
always reduce the total by moving everything else. A robust kernel caps the
influence of a large residual, which is what keeps the map standing.

Formulation
-----------
Each edge contributes a squared Mahalanobis error ``s = e^T Omega e``. A kernel
replaces that with ``rho(s)``. Iteratively reweighted least squares then solves
the ordinary problem with the information matrix scaled by the weight
``w(s) = d rho / d s``:

    H = sum_k w_k J_k^T Omega_k J_k        b = -sum_k w_k J_k^T Omega_k e_k

Every kernel below returns ``(rho, weight)`` from one call so a solver never
has to evaluate the two separately, and every kernel satisfies
``rho(s) -> s`` and ``w(s) -> 1`` as ``s -> 0`` -- inliers are untouched.

The second-order Triggs correction to the Gauss-Newton Hessian is deliberately
not applied. It can make ``H`` indefinite for redescending kernels, and the
Levenberg-Marquardt damping in :mod:`posegraph.solver` already covers the same
ground. See ``docs/theory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "CHI2_CRITICAL",
    "default_delta",
    "RobustKernel",
    "TrivialKernel",
    "HuberKernel",
    "CauchyKernel",
    "GemanMcClureKernel",
    "DCSKernel",
    "KERNELS",
    "make_kernel",
]


#: Upper-tail critical values of the chi-squared distribution, by degrees of
#: freedom. These are the standard table values; they are here so a kernel width
#: can be chosen from the noise model rather than guessed.
CHI2_CRITICAL: Dict[str, Dict[int, float]] = {
    "0.90": {1: 2.706, 2: 4.605, 3: 6.251, 4: 7.779, 5: 9.236, 6: 10.645},
    "0.95": {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592},
    "0.99": {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086, 6: 16.812},
}


def default_delta(dof: int, confidence: str = "0.95") -> float:
    """Kernel width for a ``dof``-dimensional residual.

    A kernel's ``delta`` is in units of ``sqrt(chi2)``, so the principled choice
    is the square root of the chi-squared critical value at the confidence level
    you are willing to call an inlier. For an SE(2) edge (3 degrees of freedom)
    at 95% that is ``sqrt(7.815) = 2.80``; for SE(3) (6 dof) it is
    ``sqrt(12.592) = 3.55``.

    Picking ``delta = 1`` instead -- a common default -- treats a perfectly
    ordinary residual as an outlier and quietly biases a clean graph, which is
    why this is computed rather than assumed.
    """
    table = CHI2_CRITICAL.get(confidence)
    if table is None:
        raise ValueError(f"confidence must be one of {sorted(CHI2_CRITICAL)}")
    if dof not in table:
        raise ValueError(f"no tabulated critical value for {dof} degrees of freedom")
    return float(np.sqrt(table[dof]))


@dataclass
class RobustKernel:
    """Base class. ``delta`` is the kernel width in units of ``sqrt(chi2)``."""

    delta: float = 1.0
    name: str = "trivial"

    def __post_init__(self) -> None:
        if self.delta <= 0.0:
            raise ValueError("kernel width must be positive")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rho(s), drho/ds)`` for an array of squared errors."""
        raise NotImplementedError

    def weight(self, s: np.ndarray) -> np.ndarray:
        """IRLS weight ``drho/ds``."""
        return self.evaluate(np.asarray(s, dtype=float))[1]

    def cost(self, s: np.ndarray) -> np.ndarray:
        """Robustified cost ``rho(s)``."""
        return self.evaluate(np.asarray(s, dtype=float))[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(delta={self.delta})"


class TrivialKernel(RobustKernel):
    """No robustification: ``rho(s) = s``, ``w = 1``."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(delta=delta, name="trivial")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        return s, np.ones_like(s)


class HuberKernel(RobustKernel):
    """Quadratic inside ``delta``, linear in ``|e|`` outside.

    ``rho(s) = s`` for ``s <= delta^2`` and ``2 delta sqrt(s) - delta^2``
    otherwise, so ``w(s) = delta / sqrt(s)`` for outliers. Convex, so it never
    creates new local minima -- the safe default.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(delta=delta, name="huber")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        d2 = self.delta * self.delta
        inside = s <= d2
        root = np.sqrt(np.where(inside, d2, s))
        rho = np.where(inside, s, 2.0 * self.delta * root - d2)
        w = np.where(inside, 1.0, self.delta / root)
        return rho, w


class CauchyKernel(RobustKernel):
    """``rho(s) = c^2 log(1 + s/c^2)``, ``w(s) = 1 / (1 + s/c^2)``.

    Redescending in influence but with a weight that never reaches zero, so a
    bad edge is suppressed rather than deleted.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(delta=delta, name="cauchy")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        c2 = self.delta * self.delta
        inv = 1.0 / (1.0 + s / c2)
        return c2 * np.log1p(s / c2), inv


class GemanMcClureKernel(RobustKernel):
    """``rho(s) = c^2 s / (c^2 + s)``, ``w(s) = c^4 / (c^2 + s)^2``.

    Strongly redescending: the weight falls off as ``1/s^2``, so a gross
    outlier is effectively removed. Needs a decent initial guess, because a
    residual that starts large can never climb back.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(delta=delta, name="geman_mcclure")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        c2 = self.delta * self.delta
        den = c2 + s
        return c2 * s / den, (c2 * c2) / (den * den)


class DCSKernel(RobustKernel):
    """Dynamic Covariance Scaling (Agarwal et al., ICRA 2013).

    DCS scales the information matrix of an edge by ``sigma^2`` where
    ``sigma = min(1, 2 Phi / (Phi + s))``. That is exactly an IRLS weight, so
    it fits the same interface: ``w(s) = sigma^2``. ``Phi = delta^2``.

    It behaves like Geman-McClure for large residuals but reaches full weight
    at a hard threshold instead of asymptotically, which makes it recover
    faster than a switchable-constraint formulation while costing nothing
    extra per iteration.

    The cost returned here is the integral of the weight,

        rho(s) = s                    for s <= Phi
        rho(s) = 3 Phi - 4 Phi^2 / (Phi + s)   otherwise

    rather than the ``sigma^2 s`` that the original scaling argument suggests.
    The two differ, and ``sigma^2 s`` is *not* the antiderivative of
    ``sigma^2``. Using it would make Levenberg-Marquardt compare an actual
    reduction in one function against a reduction predicted for another, and
    the gain ratio would be wrong. The form above is continuous at ``Phi``,
    saturates at ``3 Phi``, and has exactly the DCS weight as its derivative.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(delta=delta, name="dcs")

    def evaluate(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        phi = self.delta * self.delta
        inside = s <= phi
        sigma = np.minimum(1.0, 2.0 * phi / (phi + s))
        w = sigma * sigma
        rho = np.where(inside, s, 3.0 * phi - 4.0 * phi * phi / (phi + s))
        return rho, w


KERNELS: Dict[str, type] = {
    "trivial": TrivialKernel,
    "none": TrivialKernel,
    "huber": HuberKernel,
    "cauchy": CauchyKernel,
    "geman_mcclure": GemanMcClureKernel,
    "dcs": DCSKernel,
}


def make_kernel(spec: str | RobustKernel | None, delta: float = 1.0) -> RobustKernel:
    """Build a kernel from a name (``"huber"``, ``"dcs"``, ...) or pass one through."""
    if spec is None:
        return TrivialKernel()
    if isinstance(spec, RobustKernel):
        return spec
    key = str(spec).lower()
    if key not in KERNELS:
        raise ValueError(f"unknown kernel {spec!r}; choose from {sorted(KERNELS)}")
    return KERNELS[key](delta=delta)
