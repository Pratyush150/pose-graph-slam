"""All three linear-solve backends must produce the same answer."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import linalg


def _spd_system(n, rng, density=0.02):
    A = np.zeros((n, n))
    for _ in range(int(density * n * n)):
        i, j = rng.integers(0, n, 2)
        if i != j:
            v = rng.normal() * 0.2
            A[i, j] = A[j, i] = v
    A += np.eye(n) * (np.abs(A).sum(axis=1).max() + 1.0)
    x = rng.normal(size=n)
    return A, x, A @ x


def test_backends_agree(rng):
    n = 120
    A, x_true, b = _spd_system(n, rng)
    r, c = np.nonzero(A)
    data = A[r, c]
    answers = {}
    for backend in ("dense", "scipy_splu", "numpy_sparse"):
        if backend == "scipy_splu" and not linalg.HAVE_SCIPY:
            continue
        x, used = linalg.solve_spd(r, c, data, n, b, backend=backend)
        assert used == backend
        answers[backend] = x
        assert np.abs(x - x_true).max() < 1e-9
    keys = list(answers)
    for k in keys[1:]:
        assert np.abs(answers[k] - answers[keys[0]]).max() < 1e-9


def test_permutation_does_not_change_the_solution(rng):
    n = 90
    A, x_true, b = _spd_system(n, rng)
    r, c = np.nonzero(A)
    adj = linalg.block_adjacency(n, r, c)
    perm = linalg.minimum_degree_ordering(adj)
    for backend in ("dense", "numpy_sparse"):
        x, _ = linalg.solve_spd(r, c, A[r, c], n, b, perm=perm, backend=backend)
        assert np.abs(x - x_true).max() < 1e-9


def test_numpy_cholesky_rejects_indefinite_matrices():
    n = 3
    A = np.diag([1.0, -1.0, 1.0])
    r, c = np.nonzero(A)
    with pytest.raises(np.linalg.LinAlgError):
        linalg.sparse_cholesky_numpy(r, c, A[r, c], n, np.ones(n))


def test_auto_backend_picks_dense_for_small_systems(rng):
    n = 20
    A, x_true, b = _spd_system(n, rng, density=0.2)
    r, c = np.nonzero(A)
    _x, used = linalg.solve_spd(r, c, A[r, c], n, b, backend="auto")
    assert used == "dense"


def test_unknown_backend_raises(rng):
    A, _x, b = _spd_system(10, rng, density=0.3)
    r, c = np.nonzero(A)
    with pytest.raises(ValueError):
        linalg.solve_spd(r, c, A[r, c], 10, b, backend="nope")


def test_duplicate_triplets_are_summed(rng):
    """COO assembly relies on duplicate entries adding, exactly like a real Hessian."""
    n = 4
    rows = np.array([0, 0, 1, 2, 3, 1])
    cols = np.array([0, 0, 1, 2, 3, 1])
    data = np.array([1.0, 1.0, 1.0, 3.0, 4.0, 1.0])
    b = np.ones(n)
    x, _ = linalg.solve_spd(rows, cols, data, n, b, backend="dense")
    assert np.allclose(x, np.array([0.5, 0.5, 1.0 / 3.0, 0.25]))
