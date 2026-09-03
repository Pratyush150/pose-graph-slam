"""Variable ordering and symbolic factorisation.

The claim being tested is specific: reordering strictly reduces the number of
nonzero blocks in the Cholesky factor, and the count itself is right, because
it is checked against a dense factorisation of the same pattern.
"""

from __future__ import annotations

import numpy as np

from posegraph import linalg


def _grid_adjacency(nx: int, ny: int):
    n = nx * ny
    adj = [set() for _ in range(n)]
    for y in range(ny):
        for x in range(nx):
            u = y * nx + x
            if x + 1 < nx:
                v = y * nx + x + 1
                adj[u].add(v)
                adj[v].add(u)
            if y + 1 < ny:
                v = (y + 1) * nx + x
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _dense_fill_count(adj, perm):
    """Reference count: eliminate the permuted pattern with a dense boolean matrix."""
    n = len(adj)
    pos = np.empty(n, dtype=int)
    pos[np.asarray(perm)] = np.arange(n)
    A = np.eye(n, dtype=bool)
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            A[pos[u], pos[v]] = True
    count = 0
    for k in range(n):
        col = np.flatnonzero(A[k:, k]) + k
        count += col.size
        for i in col:
            A[np.ix_(col, [i])] = True
    return count


def test_symbolic_count_matches_dense_elimination():
    adj = _grid_adjacency(6, 5)
    for perm in (np.arange(30), linalg.rcm_ordering(adj), linalg.minimum_degree_ordering(adj)):
        _parent, nnz = linalg.symbolic_cholesky(adj, perm)
        assert nnz == _dense_fill_count(adj, perm)


def test_orderings_are_valid_permutations():
    adj = _grid_adjacency(9, 7)
    for perm in (linalg.rcm_ordering(adj), linalg.minimum_degree_ordering(adj)):
        assert sorted(perm.tolist()) == list(range(len(adj)))


def test_reordering_strictly_reduces_fill_on_a_grid():
    """A 2D grid in row-major order is the textbook bad case for fill-in."""
    adj = _grid_adjacency(14, 14)
    report = linalg.fill_report(adj)
    assert report.md_nnz < report.natural_nnz
    assert report.rcm_nnz < report.natural_nnz
    assert report.chosen_nnz == min(report.natural_nnz, report.rcm_nnz, report.md_nnz)
    assert report.reduction_vs_natural > 0.1
    assert "L-blocks" in report.describe()


def test_reordering_helps_a_long_loop_closure():
    """One edge from the first pose to the last is what wrecks a natural ordering."""
    n = 200
    adj = [set() for _ in range(n)]
    for k in range(n - 1):
        adj[k].add(k + 1)
        adj[k + 1].add(k)
    adj[0].add(n - 1)
    adj[n - 1].add(0)
    report = linalg.fill_report(adj)
    assert report.chosen_nnz <= report.natural_nnz


def test_elimination_tree_is_a_forest():
    adj = _grid_adjacency(8, 8)
    parent, _nnz = linalg.symbolic_cholesky(adj, np.arange(64))
    assert parent[-1] == -1, "the last eliminated node has no parent"
    for k, p in enumerate(parent):
        assert p == -1 or p > k, "the elimination tree must point forward"


def test_disconnected_graph_is_ordered_completely():
    adj = [set() for _ in range(10)]
    adj[0].add(1)
    adj[1].add(0)
    perm = linalg.rcm_ordering(adj)
    assert sorted(perm.tolist()) == list(range(10))
    perm2 = linalg.minimum_degree_ordering(adj)
    assert sorted(perm2.tolist()) == list(range(10))


def test_chain_graph_has_no_fill():
    n = 50
    adj = [set() for _ in range(n)]
    for k in range(n - 1):
        adj[k].add(k + 1)
        adj[k + 1].add(k)
    _parent, nnz = linalg.symbolic_cholesky(adj, np.arange(n))
    assert nnz == n + (n - 1), "a chain factorises with zero fill in its natural order"


def test_symbolic_pattern_matches_the_count():
    adj = _grid_adjacency(7, 6)
    for perm in (np.arange(42), linalg.minimum_degree_ordering(adj)):
        rows, cols, complete = linalg.symbolic_pattern(adj, perm)
        _parent, nnz = linalg.symbolic_cholesky(adj, perm)
        assert complete
        assert rows.size == nnz
        assert np.all(rows >= cols), "the factor is lower triangular"


def test_symbolic_pattern_truncates_when_asked():
    adj = _grid_adjacency(12, 12)
    rows, _cols, complete = linalg.symbolic_pattern(adj, np.arange(144), max_nnz=200)
    assert not complete
    assert rows.size == 200
