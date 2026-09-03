"""A deliberately minimal SLAM front end, so the package runs with no data.

This is **not** a SLAM front end in any serious sense. It has no sensor model,
no scan matching and no place recognition: it drives a synthetic trajectory,
corrupts the relative motions with Gaussian noise, and declares a loop closure
whenever two poses happen to come within a radius of each other. Its whole
purpose is to produce a pose graph with known ground truth so the back end can
be exercised and tested offline.

The real evidence for this package is in ``benchmarks/`` and runs on published
datasets. Treat anything produced here as a demo, not as a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import se2, se3
from .graph import PoseGraph

__all__ = [
    "SimulatedGraph",
    "simulate_se2",
    "simulate_se3",
    "inject_false_loop_closures",
    "perturb_poses",
]


@dataclass
class SimulatedGraph:
    """A noisy pose graph plus the ground-truth poses that generated it."""

    graph: PoseGraph
    ground_truth: np.ndarray
    num_loop_closures: int
    num_false_loop_closures: int = 0
    false_edge_indices: Optional[np.ndarray] = None


def _information(sigmas: Sequence[float]) -> np.ndarray:
    s = np.asarray(sigmas, dtype=float)
    return np.diag(1.0 / (s * s))


def simulate_se2(
    num_poses: int = 300,
    step: float = 1.0,
    turn_every: int = 25,
    turn_angle: float = np.pi / 2,
    trans_sigma: float = 0.05,
    rot_sigma: float = 0.02,
    loop_radius: float = 1.5,
    min_index_gap: int = 30,
    seed: int = 0,
) -> SimulatedGraph:
    """Drive a square-ish 2D circuit, add odometry noise and close loops.

    Returns the noisy graph (whose vertex estimates are the drifted odometry
    chain, exactly like a real front end would hand over) and the ground truth.
    """
    rng = np.random.default_rng(seed)
    gt = np.zeros((num_poses, 3))
    for k in range(1, num_poses):
        turn = turn_angle if k % turn_every == 0 else 0.0
        gt[k] = se2.compose(gt[k - 1], np.array([step, 0.0, turn]))

    info = _information([trans_sigma, trans_sigma, rot_sigma])
    g = PoseGraph(space="SE2")
    g.add_pose(0, np.zeros(3))
    est = np.zeros((num_poses, 3))
    for k in range(1, num_poses):
        true_rel = se2.between(gt[k - 1], gt[k])
        noise = np.array(
            [
                rng.normal(0.0, trans_sigma),
                rng.normal(0.0, trans_sigma),
                rng.normal(0.0, rot_sigma),
            ]
        )
        meas = se2.compose(true_rel, se2.exp(noise))
        est[k] = se2.compose(est[k - 1], meas)
        g.add_pose(k, est[k])
        g.add_edge(k - 1, k, meas, info, tag="odometry")

    n_loops = 0
    for a in range(num_poses):
        for b in range(a + min_index_gap, num_poses):
            if np.linalg.norm(gt[a, :2] - gt[b, :2]) < loop_radius:
                true_rel = se2.between(gt[a], gt[b])
                noise = np.array(
                    [
                        rng.normal(0.0, trans_sigma),
                        rng.normal(0.0, trans_sigma),
                        rng.normal(0.0, rot_sigma),
                    ]
                )
                g.add_edge(a, b, se2.compose(true_rel, se2.exp(noise)), info, tag="loop")
                n_loops += 1
                break
    g.fix_pose(0)
    return SimulatedGraph(graph=g, ground_truth=gt, num_loop_closures=n_loops)


def simulate_se3(
    num_poses: int = 200,
    step: float = 1.0,
    radius: float = 12.0,
    pitch: float = 0.6,
    trans_sigma: float = 0.05,
    rot_sigma: float = 0.02,
    loop_radius: float = 2.0,
    min_index_gap: int = 20,
    seed: int = 0,
) -> SimulatedGraph:
    """Fly a helical 3D loop, add odometry noise and close loops."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((num_poses, 7))
    gt[:, 6] = 1.0
    turns = 2.0
    for k in range(num_poses):
        t = 2.0 * np.pi * turns * k / max(num_poses - 1, 1)
        pos = np.array([radius * np.cos(t), radius * np.sin(t), pitch * t])
        yaw = t + np.pi / 2.0
        q = se3.so3_exp(np.array([0.0, 0.0, yaw]))
        gt[k] = np.concatenate([pos, np.atleast_1d(q).ravel()])

    sig = np.array([trans_sigma] * 3 + [rot_sigma] * 3)
    info = _information(sig)
    g = PoseGraph(space="SE3")
    est = np.zeros((num_poses, 7))
    est[0] = se3.identity()
    g.add_pose(0, est[0])
    for k in range(1, num_poses):
        true_rel = se3.between(gt[k - 1], gt[k])
        meas = se3.compose(true_rel, se3.exp(rng.normal(0.0, sig)))
        est[k] = se3.compose(est[k - 1], meas)
        g.add_pose(k, est[k])
        g.add_edge(k - 1, k, meas, info, tag="odometry")

    n_loops = 0
    for a in range(num_poses):
        for b in range(a + min_index_gap, num_poses):
            if np.linalg.norm(gt[a, :3] - gt[b, :3]) < loop_radius:
                true_rel = se3.between(gt[a], gt[b])
                g.add_edge(
                    a, b, se3.compose(true_rel, se3.exp(rng.normal(0.0, sig))), info, tag="loop"
                )
                n_loops += 1
                break
    g.fix_pose(0)
    return SimulatedGraph(graph=g, ground_truth=gt, num_loop_closures=n_loops)


def inject_false_loop_closures(
    graph: PoseGraph,
    count: int,
    seed: int = 0,
    min_index_gap: int = 20,
    translation_scale: float = 5.0,
) -> Tuple[PoseGraph, np.ndarray]:
    """Add ``count`` fabricated loop closures between unrelated poses.

    This is the standard way to test a robust kernel: a false positive from
    place recognition looks exactly like this -- a confident relative-pose
    measurement between two poses that were never near each other, carrying the
    same information matrix as a real one.

    Returns a copy of the graph and the indices of the injected edges.
    """
    rng = np.random.default_rng(seed)
    g = graph.copy()
    ops = g.ops
    ids = list(g.pose_ids)
    n = len(ids)
    if n < min_index_gap + 2:
        raise ValueError("graph too small to inject loop closures")
    median_info = (
        np.median(graph.edge_info, axis=0) if graph.num_edges else np.eye(g.dim)
    )
    injected = []
    attempts = 0
    while len(injected) < count and attempts < 100 * count:
        attempts += 1
        a = int(rng.integers(0, n - min_index_gap - 1))
        b = int(rng.integers(a + min_index_gap, n))
        if a == b:
            continue
        if g.space == "SE2":
            z = np.array(
                [
                    rng.normal(0.0, translation_scale),
                    rng.normal(0.0, translation_scale),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
        else:
            z = ops.exp(
                np.concatenate(
                    [rng.normal(0.0, translation_scale, 3), rng.uniform(-np.pi, np.pi, 3)]
                )
            )
        injected.append(g.num_edges)
        g.add_edge(ids[a], ids[b], z, median_info, tag="false_loop")
    return g, np.asarray(injected, dtype=np.int64)


def perturb_poses(graph: PoseGraph, sigma: float = 0.1, seed: int = 0) -> PoseGraph:
    """Randomly perturb every non-fixed pose in the tangent space."""
    rng = np.random.default_rng(seed)
    g = graph.copy()
    d = g.dim
    noise = rng.normal(0.0, sigma, size=(g.num_poses, d))
    for nid in g.fixed:
        noise[g.pose_index(nid)] = 0.0
    g.poses = np.atleast_2d(g.ops.plus(g.poses, noise))
    return g
