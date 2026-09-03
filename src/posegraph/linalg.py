"""Sparse linear algebra for the pose-graph normal equations.

Everything here works on the *block* structure of the problem: one block per
variable (3x3 for an SE(2) pose, 6x6 for an SE(3) pose, 2x2 or 3x3 for a
point). That is the structure the ordering and the symbolic analysis care
about, and it is a much smaller graph than the scalar one.

What this module provides
-------------------------
* :func:`rcm_ordering` -- reverse Cuthill-McKee, written here rather than
  imported, so the package has no hard SciPy dependency.
* :func:`minimum_degree_ordering` -- a genuine minimum-degree ordering on the
  elimination graph. It is *not* the approximate-degree (AMD) refinement; it
  computes exact degrees, which is slower to build but gives an ordering of the
  same quality.
* :func:`symbolic_cholesky` -- elimination tree and exact block nonzero count
  of the Cholesky factor, using the row-subtree algorithm. This is what the
  fill-in numbers in the README are measured with.
* :func:`solve_spd` -- solve ``H dx = b`` with a fill-reducing permutation.
  Backends, in order of preference: SciPy's sparse LU, our own pure-NumPy
  up-looking sparse Cholesky, and a dense Cholesky for small systems.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _sp = None
    _spl = None
    HAVE_SCIPY = False

__all__ = [
    "HAVE_SCIPY",
    "OrderingReport",
    "block_adjacency",
    "rcm_ordering",
    "minimum_degree_ordering",
    "symbolic_cholesky",
    "symbolic_pattern",
    "fill_report",
    "coo_to_scipy",
    "coo_to_dense",
    "solve_spd",
    "sparse_cholesky_numpy",
]


# --------------------------------------------------------------------------
# block graph helpers
# --------------------------------------------------------------------------


def block_adjacency(n: int, rows: Sequence[int], cols: Sequence[int]) -> List[set]:
    """Undirected adjacency (as sets, no self-loops) of a block graph."""
    adj: List[set] = [set() for _ in range(n)]
    for a, b in zip(rows, cols):
        a, b = int(a), int(b)
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def rcm_ordering(adj: List[set]) -> np.ndarray:
    """Reverse Cuthill-McKee ordering.

    Starts each component from a pseudo-peripheral node found by the usual
    two-sweep heuristic, then does a breadth-first sweep visiting neighbours in
    increasing degree, and finally reverses the whole thing.
    """
    n = len(adj)
    degree = np.array([len(a) for a in adj], dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    order: List[int] = []

    def bfs_levels(start: int) -> Tuple[List[int], int]:
        seen = {start}
        frontier = [start]
        levels = 0
        last = [start]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            if nxt:
                levels += 1
                last = nxt
            frontier = nxt
        return last, levels

    for seed in range(n):
        if visited[seed]:
            continue
        # pseudo-peripheral start node
        start = seed
        _, best = bfs_levels(start)
        for _ in range(3):
            last, _lv = bfs_levels(start)
            cand = min(last, key=lambda u: degree[u])
            _, lv2 = bfs_levels(cand)
            if lv2 > best:
                best, start = lv2, cand
            else:
                break
        queue = [start]
        visited[start] = True
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            order.append(u)
            nbrs = sorted((v for v in adj[u] if not visited[v]), key=lambda v: degree[v])
            for v in nbrs:
                visited[v] = True
                queue.append(v)
    return np.asarray(order[::-1], dtype=np.int64)


def minimum_degree_ordering(adj: List[set]) -> np.ndarray:
    """Minimum-degree ordering of the elimination graph.

    Repeatedly eliminates the lowest-degree vertex and turns its neighbourhood
    into a clique. A lazy heap keeps the "pick the minimum" step cheap; stale
    heap entries are discarded when popped.
    """
    n = len(adj)
    nb = [set(a) for a in adj]
    alive = np.ones(n, dtype=bool)
    heap = [(len(nb[v]), v) for v in range(n)]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        deg, v = heapq.heappop(heap)
        if not alive[v] or deg != len(nb[v]):
            continue
        alive[v] = False
        order.append(v)
        clique = nb[v]
        for u in clique:
            nb[u].discard(v)
        for u in clique:
            nb[u] |= clique
            nb[u].discard(u)
        for u in clique:
            heapq.heappush(heap, (len(nb[u]), u))
        nb[v] = set()
    # isolated vertices never entered the heap loop if n == 0
    for v in range(n):
        if alive[v]:
            order.append(v)
            alive[v] = False
    return np.asarray(order, dtype=np.int64)


def symbolic_cholesky(adj: List[set], perm: np.ndarray | None = None) -> Tuple[np.ndarray, int]:
    """Elimination tree and exact block nonzero count of ``L``.

    Uses the row-subtree algorithm: for row ``i`` walk from each earlier
    neighbour up the elimination tree until a marked node is reached. Returns
    ``(parent, nnz_blocks)`` where ``nnz_blocks`` counts the diagonal blocks as
    well.
    """
    n = len(adj)
    if perm is None:
        pos = np.arange(n)
    else:
        pos = np.empty(n, dtype=np.int64)
        pos[np.asarray(perm, dtype=np.int64)] = np.arange(n)
    # neighbours of each permuted row that come earlier
    lower: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        pu = int(pos[u])
        for v in adj[u]:
            pv = int(pos[v])
            if pv < pu:
                lower[pu].append(pv)
    parent = np.full(n, -1, dtype=np.int64)
    mark = np.full(n, -1, dtype=np.int64)
    nnz = n  # diagonal blocks
    for i in range(n):
        mark[i] = i
        for j in lower[i]:
            k = j
            while k != -1 and mark[k] != i:
                mark[k] = i
                nnz += 1
                if parent[k] == -1:
                    parent[k] = i
                k = int(parent[k])
    return parent, nnz


def symbolic_pattern(
    adj: List[set], perm: np.ndarray | None = None, max_nnz: int = 2_000_000
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Nonzero block pattern of ``L`` in permuted indices.

    Returns ``(rows, cols, complete)``. ``complete`` is False when the pattern
    was truncated at ``max_nnz``, which happens on large graphs in a bad
    ordering and is only ever used for plotting.
    """
    n = len(adj)
    if perm is None:
        pos = np.arange(n)
    else:
        pos = np.empty(n, dtype=np.int64)
        pos[np.asarray(perm, dtype=np.int64)] = np.arange(n)
    lower: List[List[int]] = [[] for _ in range(n)]
    for u in range(n):
        pu = int(pos[u])
        for v in adj[u]:
            pv = int(pos[v])
            if pv < pu:
                lower[pu].append(pv)
    parent = np.full(n, -1, dtype=np.int64)
    mark = np.full(n, -1, dtype=np.int64)
    rows: List[int] = list(range(n))
    cols: List[int] = list(range(n))
    complete = True
    for i in range(n):
        mark[i] = i
        for j in lower[i]:
            k = j
            while k != -1 and mark[k] != i:
                mark[k] = i
                if len(rows) < max_nnz:
                    rows.append(i)
                    cols.append(k)
                else:
                    complete = False
                if parent[k] == -1:
                    parent[k] = i
                k = int(parent[k])
    return np.asarray(rows), np.asarray(cols), complete


@dataclass
class OrderingReport:
    """Measured fill-in for a set of candidate orderings."""

    n_blocks: int
    natural_nnz: int
    rcm_nnz: int
    md_nnz: int
    chosen: str
    chosen_nnz: int
    perm: np.ndarray

    @property
    def reduction_vs_natural(self) -> float:
        """Fraction of the natural-ordering factor blocks that were removed."""
        if self.natural_nnz == 0:
            return 0.0
        return 1.0 - self.chosen_nnz / self.natural_nnz

    def describe(self) -> str:
        return (
            f"blocks={self.n_blocks} L-blocks natural={self.natural_nnz} "
            f"rcm={self.rcm_nnz} min-degree={self.md_nnz} "
            f"chosen={self.chosen} ({self.chosen_nnz}, "
            f"{100.0 * self.reduction_vs_natural:.1f}% fewer than natural)"
        )


def fill_report(
    adj: List[set], methods: Sequence[str] = ("natural", "rcm", "md")
) -> OrderingReport:
    """Measure factor fill-in for each ordering and keep the best."""
    n = len(adj)
    results: Dict[str, Tuple[int, np.ndarray]] = {}
    natural = np.arange(n, dtype=np.int64)
    _, nat_nnz = symbolic_cholesky(adj, natural)
    results["natural"] = (nat_nnz, natural)
    rcm_nnz = md_nnz = nat_nnz
    if "rcm" in methods:
        p = rcm_ordering(adj)
        _, rcm_nnz = symbolic_cholesky(adj, p)
        results["rcm"] = (rcm_nnz, p)
    if "md" in methods:
        p = minimum_degree_ordering(adj)
        _, md_nnz = symbolic_cholesky(adj, p)
        results["md"] = (md_nnz, p)
    chosen = min(results, key=lambda k: results[k][0])
    return OrderingReport(
        n_blocks=n,
        natural_nnz=nat_nnz,
        rcm_nnz=rcm_nnz,
        md_nnz=md_nnz,
        chosen=chosen,
        chosen_nnz=results[chosen][0],
        perm=results[chosen][1],
    )


# --------------------------------------------------------------------------
# assembly helpers
# --------------------------------------------------------------------------


def coo_to_scipy(rows, cols, data, n):
    """Build a SciPy CSC matrix from triplets (duplicates are summed)."""
    if not HAVE_SCIPY:  # pragma: no cover
        raise RuntimeError("SciPy is not available")
    return _sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()


def coo_to_dense(rows, cols, data, n) -> np.ndarray:
    """Build a dense matrix from triplets (duplicates are summed)."""
    H = np.zeros((n, n))
    np.add.at(H, (np.asarray(rows), np.asarray(cols)), np.asarray(data))
    return H


def _csc_from_coo(rows, cols, data, n):
    """Pure-NumPy CSC (indptr, indices, values) with duplicates summed."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.asarray(data, dtype=float)
    key = cols * np.int64(n) + rows
    order = np.argsort(key, kind="stable")
    key = key[order]
    vals = data[order]
    uniq_mask = np.empty(key.shape[0], dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(uniq_mask)
    summed = np.add.reduceat(vals, starts)
    ukey = key[starts]
    ucols = ukey // np.int64(n)
    urows = ukey - ucols * np.int64(n)
    indptr = np.searchsorted(ucols, np.arange(n + 1))
    return indptr.astype(np.int64), urows.astype(np.int64), summed


def sparse_cholesky_numpy(rows, cols, data, n, b: np.ndarray) -> np.ndarray:
    """Solve ``H x = b`` with an up-looking sparse Cholesky, NumPy only.

    This is the fallback used when SciPy is not installed. It is a faithful
    sparse factorisation (row subtree + sparse triangular solve), not a dense
    one in disguise, but it runs the elimination loop in Python and is therefore
    slower than SciPy for large graphs.
    """
    indptr, indices, values = _csc_from_coo(rows, cols, data, n)
    # column-wise view of the lower triangle of H
    col_rows: List[np.ndarray] = []
    col_vals: List[np.ndarray] = []
    for j in range(n):
        sl = slice(indptr[j], indptr[j + 1])
        r = indices[sl]
        v = values[sl]
        keep = r >= j
        col_rows.append(r[keep])
        col_vals.append(v[keep])

    # symbolic: adjacency of the block graph is the scalar graph here
    adj = [set() for _ in range(n)]
    for j in range(n):
        for r in col_rows[j]:
            if r != j:
                adj[int(r)].add(j)
                adj[j].add(int(r))
    parent = np.full(n, -1, dtype=np.int64)
    mark = np.full(n, -1, dtype=np.int64)

    # L stored by row while building (up-looking), then used by column
    L_rows: List[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n
    L_vals: List[np.ndarray] = [np.zeros(0)] * n
    L_diag = np.zeros(n)
    Lcol: List[List[int]] = [[] for _ in range(n)]  # rows i having L[i, j] != 0
    Lcol_val: List[List[float]] = [[] for _ in range(n)]

    x = np.zeros(n)
    # row-wise pattern of H (lower triangle, columns < i)
    row_pat: List[List[int]] = [[] for _ in range(n)]
    row_val: List[List[float]] = [[] for _ in range(n)]
    diagH = np.zeros(n)
    for j in range(n):
        for r, v in zip(col_rows[j], col_vals[j]):
            r = int(r)
            if r == j:
                diagH[j] = v
            else:
                row_pat[r].append(j)
                row_val[r].append(v)

    for i in range(n):
        # ereach: columns of L in row i, topologically ordered
        stack: List[int] = []
        mark[i] = i
        for j in row_pat[i]:
            k = j
            local: List[int] = []
            while k != -1 and mark[k] != i:
                mark[k] = i
                local.append(k)
                if parent[k] == -1:
                    parent[k] = i
                k = int(parent[k])
            stack.extend(reversed(local))
        pattern = sorted(set(stack))
        for j, v in zip(row_pat[i], row_val[i]):
            x[j] = v
        # sparse triangular solve L[0:i,0:i] y = H[i, 0:i]
        for j in pattern:
            xj = x[j] / L_diag[j]
            x[j] = xj
            if xj != 0.0:
                cr = Lcol[j]
                cv = Lcol_val[j]
                for t in range(len(cr)):
                    x[cr[t]] -= cv[t] * xj
        vals = np.array([x[j] for j in pattern])
        d = diagH[i] - float(vals @ vals)
        if d <= 0.0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L_diag[i] = np.sqrt(d)
        L_rows[i] = np.asarray(pattern, dtype=np.int64)
        L_vals[i] = vals
        for j, v in zip(pattern, vals):
            Lcol[j].append(i)
            Lcol_val[j].append(float(v))
        for j in pattern:
            x[j] = 0.0
        for j in row_pat[i]:
            x[j] = 0.0

    # forward substitution L y = b
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        if L_rows[i].size:
            s -= float(L_vals[i] @ y[L_rows[i]])
        y[i] = s / L_diag[i]
    # back substitution L^T z = y
    z = np.zeros(n)
    for i in range(n - 1, -1, -1):
        z[i] = y[i] / L_diag[i]
        if L_rows[i].size:
            y[L_rows[i]] -= L_vals[i] * z[i]
    return z


def solve_spd(
    rows,
    cols,
    data,
    n: int,
    b: np.ndarray,
    perm: np.ndarray | None = None,
    backend: str = "auto",
    dense_threshold: int = 600,
) -> Tuple[np.ndarray, str]:
    """Solve the symmetric positive-definite system ``H dx = b``.

    Returns ``(dx, backend_used)``. ``perm`` is a scalar-level fill-reducing
    permutation; the system is permuted, factorised and un-permuted.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.asarray(data, dtype=float)
    b = np.asarray(b, dtype=float)

    if perm is not None:
        perm = np.asarray(perm, dtype=np.int64)
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = np.arange(n)
        rows_p, cols_p, b_p = inv[rows], inv[cols], b[perm]
    else:
        rows_p, cols_p, b_p = rows, cols, b

    chosen = backend
    if chosen == "auto":
        if n <= dense_threshold:
            chosen = "dense"
        elif HAVE_SCIPY:
            chosen = "scipy_splu"
        else:
            chosen = "numpy_sparse"

    if chosen == "dense":
        H = coo_to_dense(rows_p, cols_p, data, n)
        L = np.linalg.cholesky(H)
        y = np.linalg.solve(L, b_p)
        x_p = np.linalg.solve(L.T, y)
    elif chosen == "scipy_splu":
        H = coo_to_scipy(rows_p, cols_p, data, n)
        lu = _spl.splu(H, permc_spec="NATURAL", diag_pivot_thresh=0.0)
        x_p = lu.solve(b_p)
    elif chosen == "numpy_sparse":
        x_p = sparse_cholesky_numpy(rows_p, cols_p, data, n, b_p)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if perm is not None:
        x = np.empty(n)
        x[perm] = x_p
        return x, chosen
    return x_p, chosen
