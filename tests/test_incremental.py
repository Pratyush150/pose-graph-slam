"""Windowed incremental updates against full batch re-optimisation."""

from __future__ import annotations

import numpy as np

from posegraph import se2
from posegraph.graph import PoseGraph
from posegraph.incremental import (
    IncrementalOptimizer,
    compare_with_batch,
    extract_subgraph,
    neighbourhood,
)
from posegraph.solver import SolverOptions, optimize


def _chain(n=60, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.zeros(3))
    info = np.eye(3) * 400.0
    for k in range(1, n):
        z = se2.compose(np.array([1.0, 0.0, 0.0]), se2.exp(rng.normal(0.0, noise, 3)))
        g.add_pose(k, se2.compose(g.pose(k - 1), z))
        g.add_edge(k - 1, k, z, info)
    g.fix_pose(0)
    return g


def test_neighbourhood_grows_with_hops():
    g = _chain(40)
    assert neighbourhood(g, [20], 0) == {20}
    assert neighbourhood(g, [20], 1) == {19, 20, 21}
    assert len(neighbourhood(g, [20], 5)) == 11


def test_subgraph_anchors_its_boundary():
    g = _chain(40)
    rows = neighbourhood(g, [20], 3)
    sub, ids, anchors = extract_subgraph(g, rows)
    assert sub.num_poses == len(rows)
    assert set(anchors) <= set(ids)
    assert len(anchors) == 2, "a chain window touches the rest of the graph at both ends"
    assert sub.num_edges == len(rows) - 1


def test_subgraph_of_the_whole_graph_keeps_the_original_anchor():
    g = _chain(20)
    sub, ids, anchors = extract_subgraph(g, set(range(g.num_poses)))
    assert anchors == [g.fixed[0]]
    assert sub.num_edges == g.num_edges


def test_incremental_update_matches_batch_when_the_window_covers_the_change():
    g = _chain(60)
    z = se2.between(g.pose(10), g.pose(40))
    z = se2.compose(z, se2.exp(np.array([0.15, -0.1, 0.02])))
    info = np.eye(3) * 400.0

    inc = g.copy()
    IncrementalOptimizer(inc, hops=60, options=SolverOptions(max_iterations=60)).add_loop_closure(
        10, 40, z, info
    )
    bat = g.copy()
    bat.add_edge(10, 40, z, info, tag="loop")
    optimize(bat, SolverOptions(max_iterations=60))

    diff = np.linalg.norm(inc.poses[:, :2] - bat.poses[:, :2], axis=1)
    assert diff.max() < 1e-6
    assert abs(inc.chi2() - bat.chi2()) < 1e-8


def test_small_window_is_cheaper_but_leaves_more_error():
    g = _chain(200)
    z = se2.compose(se2.between(g.pose(20), g.pose(150)), se2.exp(np.array([0.3, -0.2, 0.05])))
    info = np.eye(3) * 400.0

    narrow = g.copy()
    rec = IncrementalOptimizer(
        narrow, hops=5, options=SolverOptions(max_iterations=40)
    ).add_loop_closure(20, 150, z, info)
    assert rec.poses_in_window < narrow.num_poses
    assert rec.edges_in_window < narrow.num_edges
    assert rec.chi2_after < rec.chi2_before
    assert "window" in rec.describe()

    wide = g.copy()
    IncrementalOptimizer(wide, hops=500, options=SolverOptions(max_iterations=40)).add_loop_closure(
        20, 150, z, info
    )
    assert wide.chi2() <= rec.chi2_after + 1e-9


def test_add_odometry_extends_the_graph_without_solving():
    g = _chain(10)
    inc = IncrementalOptimizer(g, hops=3)
    prev = g.pose(9)
    z = np.array([1.0, 0.0, 0.0])
    inc.add_odometry(9, 10, z, np.eye(3) * 100.0)
    assert g.num_poses == 11
    assert g.num_edges == 10
    assert np.allclose(g.pose(10), se2.compose(prev, z), atol=1e-12)
    assert g.edge_chi2()[-1] < 1e-20, "chaining the estimate makes the new edge exact"
    assert inc.records == []


def test_batch_update_records_timing():
    g = _chain(30)
    inc = IncrementalOptimizer(g, hops=3, options=SolverOptions(max_iterations=20))
    rec = inc.batch()
    assert rec.seconds > 0.0
    assert rec.poses_in_window == g.num_poses
    assert inc.total_seconds() > 0.0


def test_compare_with_batch_reports_both():
    g = _chain(120)
    info = np.eye(3) * 400.0
    closures = []
    for a, b in ((10, 60), (20, 90), (30, 110)):
        z = se2.compose(se2.between(g.pose(a), g.pose(b)), se2.exp(np.array([0.1, 0.05, 0.01])))
        closures.append((a, b, z, info))
    out = compare_with_batch(g, closures, hops=200, options=SolverOptions(max_iterations=40))
    assert len(out["incremental_times"]) == 3
    assert len(out["batch_times"]) == 3
    assert out["max_position_difference"] < 1e-4
    assert out["incremental_mean_ms"] > 0.0
    assert out["batch_mean_ms"] > 0.0
