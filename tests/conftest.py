"""Shared fixtures. Adds ``src/`` to the path so plain ``pytest`` works uninstalled."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data"


def dataset_available(name: str) -> bool:
    """True when a benchmark file has been fetched into ``data/``."""
    from posegraph import datasets

    try:
        return datasets.dataset_path(name, DATA_DIR).exists()
    except KeyError:
        return False


def requires_dataset(name: str):
    """Skip (never fail) a test whose benchmark file has not been downloaded."""
    return pytest.mark.skipif(
        not dataset_available(name),
        reason=f"benchmark dataset {name!r} not downloaded; "
        f"run tools/fetch_datasets.py {name}",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator: every test in this suite is reproducible."""
    return np.random.default_rng(20260101)


def numerical_jacobian(f, x, plus, dof, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``f`` at ``x`` under the retraction ``plus``."""
    y0 = np.atleast_1d(f(x))
    J = np.zeros((y0.size, dof))
    for k in range(dof):
        dp = np.zeros(dof)
        dm = np.zeros(dof)
        dp[k] = eps
        dm[k] = -eps
        J[:, k] = (np.atleast_1d(f(plus(x, dp))) - np.atleast_1d(f(plus(x, dm)))) / (2.0 * eps)
    return J
