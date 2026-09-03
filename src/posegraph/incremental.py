"""Incremental optimisation: re-solve only the part of the graph that moved.

The observation is simple. When a loop closure arrives between poses ``a`` and
``b``, the correction it implies is large near ``a`` and ``b`` and decays away
from them, because the rest of the graph is already internally consistent.
Re-factorising the whole information matrix to discover that most of the
solution did not change is wasted work.

What this module does is the cheap version of that idea: take every pose within
``hops`` edges of the new constraint, hold the boundary of that neighbourhood
fixed, and optimise the interior. The boundary poses act as anchors, so the
sub-problem is gauge-fixed by construction and the correction cannot leak into
parts of the map that had no reason to move.

What it is not: this is not iSAM2. There is no Bayes tree, no incremental
re-ordering, and no fluid relinearisation. It is a windowed batch solve. The
honest trade-off is measured, not asserted -- :func:`compare_with_batch` runs
both and reports the time and the pose difference, and the tests assert that
the incremental result matches the batch result to a stated tolerance when the
window is large enough to contain the correction.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .graph import PoseGraph
from .solver import SolverOptions, optimize

__all__ = [
    "UpdateRecord",
    "IncrementalOptimizer",
    "neighbourhood",
    "extract_subgraph",
    "compare_with_batch",
]


@dataclass
class UpdateRecord:
    """One incremental update, with everything needed to compare against batch."""

    trigger: str
    poses_in_window: int
    edges_in_window: int
    anchors: int
    iterations: int
    seconds: float
    chi2_before: float
    chi2_after: float

    def describe(self) -> str:
        return (
            f"{self.trigger}: window {self.poses_in_window} poses / "
            f"{self.edges_in_window} edges ({self.anchors} anchored), "
            f"{self.iterations} iters, {1e3 * self.seconds:.1f} ms, "
            f"chi2 {self.chi2_before:.6g} -> {self.chi2_after:.6g}"
        )


def neighbourhood(graph: PoseGraph, seeds: Sequence[int], hops: int) -> Set[int]:
    """Pose *rows* within ``hops`` edges of any seed row (seeds included)."""
    adj = graph.adjacency()
    seen: Set[int] = set(int(s) for s in seeds)
    frontier = deque((int(s), 0) for s in seeds)
    while frontier:
        u, depth = frontier.popleft()
        if depth >= hops:
            continue
        for v, _k, _f in adj[u]:
            if v not in seen:
                seen.add(v)
                frontier.append((v, depth + 1))
    return seen


def extract_subgraph(
    graph: PoseGraph, rows: Set[int]
) -> Tuple[PoseGraph, List[int], List[int]]:
    """Build the induced subgraph over ``rows``.

    Returns ``(subgraph, node_ids, anchor_ids)``. Any pose in ``rows`` that has
    an edge leaving the set is added to the subgraph's fixed set, which both
    anchors the window in the global frame and removes its gauge freedom.
    """
    ids = [graph.pose_ids[r] for r in sorted(rows)]
    id_set = set(ids)
    sub = PoseGraph(space=graph.space)
    for nid in ids:
        sub.add_pose(nid, graph.pose(nid))

    ri, rj = graph.edge_rows()
    inside = np.zeros(graph.num_poses, dtype=bool)
    inside[list(rows)] = True
    both = inside[ri] & inside[rj]
    crossing = inside[ri] ^ inside[rj]

    keep = np.flatnonzero(both)
    if keep.size:
        sub.edge_i = graph.edge_i[keep].copy()
        sub.edge_j = graph.edge_j[keep].copy()
        sub.edge_z = graph.edge_z[keep].copy()
        sub.edge_info = graph.edge_info[keep].copy()
        sub.edge_tag = [graph.edge_tag[k] for k in keep]

    anchors: List[int] = []
    cross_idx = np.flatnonzero(crossing)
    boundary_rows = set(int(r) for r in ri[cross_idx] if inside[r]) | set(
        int(r) for r in rj[cross_idx] if inside[r]
    )
    for r in sorted(boundary_rows):
        anchors.append(graph.pose_ids[r])
    for nid in graph.fixed:
        if nid in id_set and nid not in anchors:
            anchors.append(nid)
    if not anchors and ids:
        anchors.append(ids[0])
    for nid in anchors:
        sub.fix_pose(nid)
    return sub, ids, anchors


class IncrementalOptimizer:
    """Keeps a pose graph optimised as constraints arrive.

    Parameters
    ----------
    graph:
        The graph to maintain. It is optimised in place.
    hops:
        Radius of the re-optimisation window, in edges.
    options:
        Solver options used for both window and batch solves.
    """

    def __init__(
        self,
        graph: PoseGraph,
        hops: int = 4,
        options: SolverOptions | None = None,
    ) -> None:
        self.graph = graph
        self.hops = int(hops)
        self.options = options or SolverOptions(method="lm", max_iterations=30)
        self.records: List[UpdateRecord] = []

    # -- graph growth -------------------------------------------------------

    def add_odometry(
        self,
        prev_id: int,
        new_id: int,
        measurement: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        """Extend the trajectory. Cheap: the new pose is chained, nothing is solved."""
        ops = self.graph.ops
        pose = ops.compose(self.graph.pose(prev_id), np.asarray(measurement, dtype=float))
        self.graph.add_pose(new_id, np.atleast_1d(pose).ravel())
        self.graph.add_edge(prev_id, new_id, measurement, information, tag="odometry")

    def add_loop_closure(
        self,
        i: int,
        j: int,
        measurement: np.ndarray,
        information: np.ndarray | None = None,
        hops: int | None = None,
    ) -> UpdateRecord:
        """Add a loop closure and re-optimise only its neighbourhood."""
        self.graph.add_edge(i, j, measurement, information, tag="loop")
        return self.update(
            [self.graph.pose_index(i), self.graph.pose_index(j)],
            hops=hops,
            trigger=f"loop {i}-{j}",
        )

    # -- solving ------------------------------------------------------------

    def update(
        self,
        seed_rows: Sequence[int],
        hops: int | None = None,
        trigger: str = "update",
    ) -> UpdateRecord:
        """Re-optimise the window around ``seed_rows`` and write the result back."""
        t0 = time.perf_counter()
        chi2_before = self.graph.chi2()
        rows = neighbourhood(self.graph, seed_rows, self.hops if hops is None else hops)
        sub, ids, anchors = extract_subgraph(self.graph, rows)
        result = optimize(sub, self.options)
        free = [nid for nid in ids if nid not in set(anchors)]
        for nid in free:
            self.graph.set_pose(nid, sub.pose(nid))
        rec = UpdateRecord(
            trigger=trigger,
            poses_in_window=len(ids),
            edges_in_window=sub.num_edges,
            anchors=len(anchors),
            iterations=result.iterations,
            seconds=time.perf_counter() - t0,
            chi2_before=chi2_before,
            chi2_after=self.graph.chi2(),
        )
        self.records.append(rec)
        return rec

    def batch(self, trigger: str = "batch") -> UpdateRecord:
        """Full batch re-optimisation of the whole graph."""
        t0 = time.perf_counter()
        chi2_before = self.graph.chi2()
        if not self.graph.fixed and not self.graph.priors:
            self.graph.fix_pose(self.graph.pose_ids[0])
        result = optimize(self.graph, self.options)
        rec = UpdateRecord(
            trigger=trigger,
            poses_in_window=self.graph.num_poses,
            edges_in_window=self.graph.num_edges,
            anchors=len(self.graph.fixed),
            iterations=result.iterations,
            seconds=time.perf_counter() - t0,
            chi2_before=chi2_before,
            chi2_after=self.graph.chi2(),
        )
        self.records.append(rec)
        return rec

    def total_seconds(self) -> float:
        return float(sum(r.seconds for r in self.records))


def compare_with_batch(
    graph: PoseGraph,
    loop_closures: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
    hops: int = 4,
    options: SolverOptions | None = None,
) -> Dict[str, object]:
    """Replay the same loop closures incrementally and in batch, and measure both.

    Returns a dict with per-update timings for the incremental run, the batch
    timings, and the largest per-pose translation difference between the two
    final trajectories.
    """
    opts = options or SolverOptions(method="lm", max_iterations=30)

    inc_graph = graph.copy()
    inc = IncrementalOptimizer(inc_graph, hops=hops, options=opts)
    inc_times: List[float] = []
    for i, j, z, info in loop_closures:
        rec = inc.add_loop_closure(i, j, z, info)
        inc_times.append(rec.seconds)

    bat_graph = graph.copy()
    bat = IncrementalOptimizer(bat_graph, hops=hops, options=opts)
    bat_times: List[float] = []
    for i, j, z, info in loop_closures:
        bat_graph.add_edge(i, j, z, info, tag="loop")
        rec = bat.batch()
        bat_times.append(rec.seconds)

    d = 2 if graph.space == "SE2" else 3
    diff = np.linalg.norm(inc_graph.poses[:, :d] - bat_graph.poses[:, :d], axis=1)
    return {
        "incremental_times": inc_times,
        "batch_times": bat_times,
        "incremental_total": float(sum(inc_times)),
        "batch_total": float(sum(bat_times)),
        "incremental_mean_ms": 1e3 * float(np.mean(inc_times)) if inc_times else 0.0,
        "batch_mean_ms": 1e3 * float(np.mean(bat_times)) if bat_times else 0.0,
        "max_position_difference": float(diff.max()) if diff.size else 0.0,
        "rms_position_difference": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
        "incremental_chi2": float(inc_graph.chi2()),
        "batch_chi2": float(bat_graph.chi2()),
        "windows": [r.poses_in_window for r in inc.records],
    }
