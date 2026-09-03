"""Pose-graph container plus a reader/writer for the ``.g2o`` file format.

A :class:`PoseGraph` holds

* pose variables (SE(2) or SE(3)), stored as one dense array;
* optional point/landmark variables (2D or 3D);
* binary relative-pose edges with information matrices;
* unary prior edges;
* a set of hard-fixed variables, which is how gauge freedom is removed.

The ``.g2o`` reader understands ``VERTEX_SE2``/``EDGE_SE2`` and
``VERTEX_SE3:QUAT``/``EDGE_SE3:QUAT`` (the two formats the public pose-graph
benchmarks ship in), plus ``VERTEX_XY``/``EDGE_SE2_XY``,
``VERTEX_TRACKXYZ``/``EDGE_SE3_TRACKXYZ`` and ``FIX``. Lines it does not
understand are kept verbatim so that read -> write is lossless.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import se2, se3

__all__ = [
    "PoseGraph",
    "PriorEdge",
    "read_g2o",
    "write_g2o",
]


def _ops(space: str):
    return se2 if space == "SE2" else se3


def _pose_width(space: str) -> int:
    return 3 if space == "SE2" else 7


def _point_width(space: str) -> int:
    return 2 if space == "SE2" else 3


@dataclass
class PriorEdge:
    """A unary measurement pinning one pose towards ``measurement``."""

    node: int
    measurement: np.ndarray
    information: np.ndarray


@dataclass
class PoseGraph:
    """A pose graph over SE(2) or SE(3) variables.

    Parameters
    ----------
    space:
        ``"SE2"`` or ``"SE3"``.
    """

    space: str = "SE2"

    pose_ids: List[int] = field(default_factory=list)
    poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    point_ids: List[int] = field(default_factory=list)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    edge_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    edge_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    edge_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    edge_info: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    edge_tag: List[str] = field(default_factory=list)

    lm_pose: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lm_point: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lm_z: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    lm_info: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))

    priors: List[PriorEdge] = field(default_factory=list)
    fixed: List[int] = field(default_factory=list)
    trailing_lines: List[str] = field(default_factory=list)

    _pose_index: Dict[int, int] = field(default_factory=dict, repr=False)
    _point_index: Dict[int, int] = field(default_factory=dict, repr=False)

    # -- construction -------------------------------------------------------

    def __post_init__(self) -> None:
        if self.space not in ("SE2", "SE3"):
            raise ValueError("space must be 'SE2' or 'SE3'")
        pw, qw = _pose_width(self.space), _point_width(self.space)
        d = self.dim
        if self.poses.shape[1] != pw:
            self.poses = np.zeros((0, pw))
        if self.points.shape[1] != qw:
            self.points = np.zeros((0, qw))
        if self.edge_z.shape[1] != pw:
            self.edge_z = np.zeros((0, pw))
            self.edge_info = np.zeros((0, d, d))
        if self.lm_z.shape[1] != qw:
            self.lm_z = np.zeros((0, qw))
            self.lm_info = np.zeros((0, qw, qw))
        if not self._pose_index:
            self._pose_index = {i: k for k, i in enumerate(self.pose_ids)}
        if not self._point_index:
            self._point_index = {i: k for k, i in enumerate(self.point_ids)}

    @property
    def ops(self):
        """The Lie-group module (:mod:`posegraph.se2` or :mod:`posegraph.se3`)."""
        return _ops(self.space)

    @property
    def dim(self) -> int:
        """Tangent-space dimension of a pose variable (3 for SE(2), 6 for SE(3))."""
        return 3 if self.space == "SE2" else 6

    @property
    def point_dim(self) -> int:
        """Dimension of a point variable (2 for SE(2), 3 for SE(3))."""
        return _point_width(self.space)

    @property
    def num_poses(self) -> int:
        return len(self.pose_ids)

    @property
    def num_points(self) -> int:
        return len(self.point_ids)

    @property
    def num_edges(self) -> int:
        """Number of binary relative-pose edges."""
        return int(self.edge_i.shape[0])

    @property
    def num_landmark_edges(self) -> int:
        return int(self.lm_pose.shape[0])

    def pose_index(self, node_id: int) -> int:
        """Row of ``node_id`` inside :attr:`poses`."""
        return self._pose_index[node_id]

    def add_pose(self, node_id: int, pose: Sequence[float] | None = None) -> int:
        """Add (or overwrite) a pose vertex. Returns its row index."""
        p = (
            np.asarray(pose, dtype=float)
            if pose is not None
            else self.ops.identity()
        )
        if p.shape != (_pose_width(self.space),):
            raise ValueError(f"pose must have shape ({_pose_width(self.space)},)")
        if node_id in self._pose_index:
            k = self._pose_index[node_id]
            self.poses[k] = p
            return k
        k = len(self.pose_ids)
        self.pose_ids.append(int(node_id))
        self._pose_index[int(node_id)] = k
        self.poses = (
            np.vstack([self.poses, p[None, :]]) if self.poses.size else p[None, :].copy()
        )
        return k

    def add_point(self, point_id: int, position: Sequence[float] | None = None) -> int:
        """Add (or overwrite) a point/landmark vertex."""
        qw = _point_width(self.space)
        p = np.asarray(position, dtype=float) if position is not None else np.zeros(qw)
        if p.shape != (qw,):
            raise ValueError(f"point must have shape ({qw},)")
        if point_id in self._point_index:
            k = self._point_index[point_id]
            self.points[k] = p
            return k
        k = len(self.point_ids)
        self.point_ids.append(int(point_id))
        self._point_index[int(point_id)] = k
        self.points = (
            np.vstack([self.points, p[None, :]]) if self.points.size else p[None, :].copy()
        )
        return k

    def add_edge(
        self,
        i: int,
        j: int,
        measurement: Sequence[float],
        information: np.ndarray | None = None,
        tag: str = "",
    ) -> None:
        """Add a binary relative-pose edge ``i -> j``.

        ``measurement`` is the pose of ``j`` expressed in the frame of ``i``.
        ``information`` defaults to the identity.
        """
        d = self.dim
        z = np.asarray(measurement, dtype=float)
        if z.shape != (_pose_width(self.space),):
            raise ValueError("measurement has the wrong shape")
        omega = np.eye(d) if information is None else np.asarray(information, dtype=float)
        if omega.shape != (d, d):
            raise ValueError(f"information must be {d}x{d}")
        for nid in (i, j):
            if nid not in self._pose_index:
                self.add_pose(nid)
        self.edge_i = np.append(self.edge_i, np.int64(i))
        self.edge_j = np.append(self.edge_j, np.int64(j))
        self.edge_z = np.vstack([self.edge_z, z[None, :]])
        self.edge_info = np.concatenate([self.edge_info, omega[None, :, :]], axis=0)
        self.edge_tag.append(tag or ("odometry" if abs(int(j) - int(i)) == 1 else "loop"))

    def add_landmark_edge(
        self,
        pose_id: int,
        point_id: int,
        measurement: Sequence[float],
        information: np.ndarray | None = None,
    ) -> None:
        """Add a pose-to-point edge: the point seen in the pose's own frame."""
        qw = _point_width(self.space)
        z = np.asarray(measurement, dtype=float)
        if z.shape != (qw,):
            raise ValueError("landmark measurement has the wrong shape")
        omega = np.eye(qw) if information is None else np.asarray(information, dtype=float)
        if pose_id not in self._pose_index:
            self.add_pose(pose_id)
        if point_id not in self._point_index:
            self.add_point(point_id)
        self.lm_pose = np.append(self.lm_pose, np.int64(pose_id))
        self.lm_point = np.append(self.lm_point, np.int64(point_id))
        self.lm_z = np.vstack([self.lm_z, z[None, :]]) if self.lm_z.size else z[None, :].copy()
        self.lm_info = np.concatenate([self.lm_info, omega[None, :, :]], axis=0)

    def add_prior(
        self, node_id: int, measurement: Sequence[float], information: np.ndarray
    ) -> None:
        """Pin ``node_id`` softly towards ``measurement``."""
        self.priors.append(
            PriorEdge(
                int(node_id),
                np.asarray(measurement, dtype=float),
                np.asarray(information, dtype=float),
            )
        )

    def fix_pose(self, node_id: int) -> None:
        """Hard-fix a pose. At least one fixed pose (or prior) is required."""
        if node_id not in self._pose_index:
            raise KeyError(f"unknown pose {node_id}")
        if node_id not in self.fixed:
            self.fixed.append(int(node_id))

    def pose(self, node_id: int) -> np.ndarray:
        return self.poses[self._pose_index[node_id]].copy()

    def set_pose(self, node_id: int, value: Sequence[float]) -> None:
        self.poses[self._pose_index[node_id]] = np.asarray(value, dtype=float)

    def copy(self) -> "PoseGraph":
        """Deep copy (arrays are copied, so optimising the copy is safe)."""
        g = PoseGraph(space=self.space)
        g.pose_ids = list(self.pose_ids)
        g.poses = self.poses.copy()
        g.point_ids = list(self.point_ids)
        g.points = self.points.copy()
        g.edge_i = self.edge_i.copy()
        g.edge_j = self.edge_j.copy()
        g.edge_z = self.edge_z.copy()
        g.edge_info = self.edge_info.copy()
        g.edge_tag = list(self.edge_tag)
        g.lm_pose = self.lm_pose.copy()
        g.lm_point = self.lm_point.copy()
        g.lm_z = self.lm_z.copy()
        g.lm_info = self.lm_info.copy()
        g.priors = [
            PriorEdge(p.node, p.measurement.copy(), p.information.copy()) for p in self.priors
        ]
        g.fixed = list(self.fixed)
        g.trailing_lines = list(self.trailing_lines)
        g._pose_index = dict(self._pose_index)
        g._point_index = dict(self._point_index)
        return g

    # -- residuals ----------------------------------------------------------

    def edge_rows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Row indices of the endpoints of every binary edge (cached)."""
        key = (self.num_edges, self.num_poses)
        cached = getattr(self, "_edge_rows_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        idx = self._pose_index
        ri = np.fromiter((idx[int(i)] for i in self.edge_i), dtype=np.int64, count=self.num_edges)
        rj = np.fromiter((idx[int(j)] for j in self.edge_j), dtype=np.int64, count=self.num_edges)
        self._edge_rows_cache = (key, ri, rj)
        return ri, rj

    def edge_errors(self) -> np.ndarray:
        """Stack of binary-edge errors, shape ``(num_edges, dim)``."""
        if self.num_edges == 0:
            return np.zeros((0, self.dim))
        ri, rj = self.edge_rows()
        return np.atleast_2d(
            self.ops.relative_error(self.poses[ri], self.poses[rj], self.edge_z)
        )

    def landmark_errors(self) -> np.ndarray:
        """Stack of landmark-edge errors, shape ``(num_landmark_edges, point_dim)``."""
        if self.num_landmark_edges == 0:
            return np.zeros((0, self.point_dim))
        pr = np.fromiter(
            (self._pose_index[int(i)] for i in self.lm_pose),
            dtype=np.int64,
            count=self.num_landmark_edges,
        )
        qr = np.fromiter(
            (self._point_index[int(i)] for i in self.lm_point),
            dtype=np.int64,
            count=self.num_landmark_edges,
        )
        X = self.poses[pr]
        P = self.points[qr]
        if self.space == "SE2":
            c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
            dx = P[:, 0] - X[:, 0]
            dy = P[:, 1] - X[:, 1]
            pred = np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
        else:
            R = np.asarray(se3.quat_to_rotation(X[:, 3:])).reshape(-1, 3, 3)
            pred = np.einsum("nji,nj->ni", R, P - X[:, :3])
        return pred - self.lm_z

    def prior_errors(self) -> List[np.ndarray]:
        """Error vector of every prior edge."""
        out = []
        for p in self.priors:
            X = self.poses[self._pose_index[p.node]]
            out.append(np.atleast_1d(self.ops.log(self.ops.between(p.measurement, X))))
        return out

    def chi2(self) -> float:
        """Total squared Mahalanobis error over all edges (no robust kernel)."""
        total = 0.0
        e = self.edge_errors()
        if e.size:
            total += float(np.einsum("ni,nij,nj->", e, self.edge_info, e))
        le = self.landmark_errors()
        if le.size:
            total += float(np.einsum("ni,nij,nj->", le, self.lm_info, le))
        for pe, p in zip(self.prior_errors(), self.priors):
            total += float(pe @ p.information @ pe)
        return total

    def edge_chi2(self) -> np.ndarray:
        """Per-edge squared Mahalanobis error of the binary edges."""
        e = self.edge_errors()
        if e.size == 0:
            return np.zeros(0)
        return np.einsum("ni,nij,nj->n", e, self.edge_info, e)

    # -- topology -----------------------------------------------------------

    def adjacency(self) -> List[List[Tuple[int, int, bool]]]:
        """Undirected adjacency list over pose rows.

        Each entry is ``(neighbour_row, edge_index, forward)`` where ``forward``
        means the stored measurement points from this row to the neighbour.
        """
        adj: List[List[Tuple[int, int, bool]]] = [[] for _ in range(self.num_poses)]
        ri, rj = self.edge_rows()
        for k in range(self.num_edges):
            adj[ri[k]].append((int(rj[k]), k, True))
            adj[rj[k]].append((int(ri[k]), k, False))
        return adj

    def initialize_from_spanning_tree(self, root: int | None = None) -> None:
        """Chain measurements along a breadth-first spanning tree.

        Benchmark ``.g2o`` files such as ``manhattan.g2o`` ship edges only. This
        walks the graph from ``root`` (default: the first pose) preferring
        short-index edges, which follows odometry first and gives the classic
        drifting "spaghetti" initial guess.
        """
        if self.num_poses == 0:
            return
        ops = self.ops
        adj = self.adjacency()
        for u, lst in enumerate(adj):
            lst.sort(key=lambda t, u=u: abs(t[0] - u))
        root_row = 0 if root is None else self._pose_index[root]
        visited = np.zeros(self.num_poses, dtype=bool)
        order = [root_row]
        visited[root_row] = True
        self.poses[root_row] = ops.identity()
        dq = deque(order)
        while dq:
            u = dq.popleft()
            for v, k, forward in adj[u]:
                if visited[v]:
                    continue
                z = self.edge_z[k]
                self.poses[v] = (
                    ops.compose(self.poses[u], z)
                    if forward
                    else ops.compose(self.poses[u], ops.inverse(z))
                )
                visited[v] = True
                dq.append(v)
        # Disconnected components keep whatever they had; report them honestly.
        if not visited.all():
            missing = int((~visited).sum())
            self.trailing_lines.append(
                f"# note: {missing} pose(s) not reachable from the spanning-tree root"
            )

    def connected_components(self) -> List[List[int]]:
        """Connected components as lists of pose rows."""
        adj = self.adjacency()
        seen = np.zeros(self.num_poses, dtype=bool)
        comps: List[List[int]] = []
        for start in range(self.num_poses):
            if seen[start]:
                continue
            comp = [start]
            seen[start] = True
            dq = deque([start])
            while dq:
                u = dq.popleft()
                for v, _k, _f in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        comp.append(v)
                        dq.append(v)
            comps.append(comp)
        return comps

    def loop_edges(self) -> np.ndarray:
        """Indices of edges that are not sequential odometry."""
        return np.flatnonzero(np.abs(self.edge_j - self.edge_i) != 1)

    def summary(self) -> str:
        return (
            f"PoseGraph({self.space}) poses={self.num_poses} points={self.num_points} "
            f"edges={self.num_edges} landmark_edges={self.num_landmark_edges} "
            f"priors={len(self.priors)} fixed={len(self.fixed)}"
        )


# --------------------------------------------------------------------------
# .g2o input / output
# --------------------------------------------------------------------------

_SE2_INFO_ORDER = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _upper_indices(d: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(d) for c in range(r, d)]


def _info_from_upper(values: Sequence[float], d: int) -> np.ndarray:
    idx = _upper_indices(d)
    if len(values) != len(idx):
        raise ValueError(f"expected {len(idx)} information entries, got {len(values)}")
    M = np.zeros((d, d))
    for (r, c), v in zip(idx, values):
        M[r, c] = v
        M[c, r] = v
    return M


def _upper_from_info(M: np.ndarray) -> List[float]:
    return [float(M[r, c]) for r, c in _upper_indices(M.shape[0])]


def _fmt(x: float) -> str:
    return repr(float(x))


def read_g2o(path: str | Path, space: str | None = None) -> PoseGraph:
    """Read a ``.g2o`` file into a :class:`PoseGraph`.

    The pose space is auto-detected from the first vertex or edge token unless
    ``space`` is given. Vertices missing from the file (some benchmarks ship
    edges only) are created and then initialised with
    :meth:`PoseGraph.initialize_from_spanning_tree`.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    detected = space
    if detected is None:
        for ln in lines:
            tok = ln.split(None, 1)[0] if ln.strip() else ""
            if tok in ("VERTEX_SE2", "EDGE_SE2", "VERTEX_XY", "EDGE_SE2_XY"):
                detected = "SE2"
                break
            if tok in (
                "VERTEX_SE3:QUAT",
                "EDGE_SE3:QUAT",
                "VERTEX_TRACKXYZ",
                "EDGE_SE3_TRACKXYZ",
            ):
                detected = "SE3"
                break
    if detected is None:
        raise ValueError(f"{path}: no recognised g2o vertex or edge records")

    g = PoseGraph(space=detected)
    d = g.dim
    have_vertex = False
    pose_rows: List[np.ndarray] = []
    e_i: List[int] = []
    e_j: List[int] = []
    e_z: List[np.ndarray] = []
    e_info: List[np.ndarray] = []
    lm_p: List[int] = []
    lm_q: List[int] = []
    lm_z: List[np.ndarray] = []
    lm_info: List[np.ndarray] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        tok = parts[0]
        try:
            if tok == "VERTEX_SE2":
                nid = int(parts[1])
                g.pose_ids.append(nid)
                g._pose_index[nid] = len(pose_rows)
                pose_rows.append(np.array([float(parts[2]), float(parts[3]), float(parts[4])]))
                have_vertex = True
            elif tok == "VERTEX_SE3:QUAT":
                nid = int(parts[1])
                g.pose_ids.append(nid)
                g._pose_index[nid] = len(pose_rows)
                v = np.array([float(x) for x in parts[2:9]])
                v[3:] = se3.quat_normalize(v[3:])
                pose_rows.append(v)
                have_vertex = True
            elif tok == "EDGE_SE2":
                e_i.append(int(parts[1]))
                e_j.append(int(parts[2]))
                e_z.append(np.array([float(parts[3]), float(parts[4]), float(parts[5])]))
                e_info.append(_info_from_upper([float(x) for x in parts[6:12]], 3))
            elif tok == "EDGE_SE3:QUAT":
                e_i.append(int(parts[1]))
                e_j.append(int(parts[2]))
                z = np.array([float(x) for x in parts[3:10]])
                z[3:] = se3.quat_normalize(z[3:])
                e_z.append(z)
                e_info.append(_info_from_upper([float(x) for x in parts[10:31]], 6))
            elif tok == "VERTEX_XY":
                pid = int(parts[1])
                g.point_ids.append(pid)
                g._point_index[pid] = len(g.point_ids) - 1
                g.points = np.vstack(
                    [g.points, np.array([[float(parts[2]), float(parts[3])]])]
                )
            elif tok == "VERTEX_TRACKXYZ":
                pid = int(parts[1])
                g.point_ids.append(pid)
                g._point_index[pid] = len(g.point_ids) - 1
                g.points = np.vstack(
                    [g.points, np.array([[float(parts[2]), float(parts[3]), float(parts[4])]])]
                )
            elif tok == "EDGE_SE2_XY":
                lm_p.append(int(parts[1]))
                lm_q.append(int(parts[2]))
                lm_z.append(np.array([float(parts[3]), float(parts[4])]))
                lm_info.append(_info_from_upper([float(x) for x in parts[5:8]], 2))
            elif tok == "EDGE_SE3_TRACKXYZ":
                lm_p.append(int(parts[1]))
                lm_q.append(int(parts[2]))
                lm_z.append(np.array([float(x) for x in parts[4:7]]))
                lm_info.append(_info_from_upper([float(x) for x in parts[7:13]], 3))
            elif tok == "FIX":
                g.fixed.append(int(parts[1]))
            else:
                g.trailing_lines.append(s)
        except (IndexError, ValueError) as exc:  # malformed record
            raise ValueError(f"{path}: cannot parse line {s!r}: {exc}") from exc

    pw = _pose_width(detected)
    if pose_rows:
        g.poses = np.asarray(pose_rows, dtype=float)
    else:
        g.poses = np.zeros((0, pw))

    # Create vertices referenced only by edges.
    for nid in list(e_i) + list(e_j) + list(lm_p):
        if nid not in g._pose_index:
            g._pose_index[nid] = len(g.pose_ids)
            g.pose_ids.append(nid)
            g.poses = np.vstack([g.poses, _ops(detected).identity()[None, :]])
    for pid in lm_q:
        if pid not in g._point_index:
            g._point_index[pid] = len(g.point_ids)
            g.point_ids.append(pid)
            g.points = np.vstack([g.points, np.zeros((1, _point_width(detected)))])

    g.edge_i = np.asarray(e_i, dtype=np.int64)
    g.edge_j = np.asarray(e_j, dtype=np.int64)
    g.edge_z = np.asarray(e_z, dtype=float) if e_z else np.zeros((0, pw))
    g.edge_info = np.asarray(e_info, dtype=float) if e_info else np.zeros((0, d, d))
    g.edge_tag = ["odometry" if abs(b - a) == 1 else "loop" for a, b in zip(e_i, e_j)]
    qw = _point_width(detected)
    g.lm_pose = np.asarray(lm_p, dtype=np.int64)
    g.lm_point = np.asarray(lm_q, dtype=np.int64)
    g.lm_z = np.asarray(lm_z, dtype=float) if lm_z else np.zeros((0, qw))
    g.lm_info = np.asarray(lm_info, dtype=float) if lm_info else np.zeros((0, qw, qw))

    if not have_vertex and g.num_edges:
        g.initialize_from_spanning_tree()
    return g


def write_g2o(graph: PoseGraph, path: str | Path) -> None:
    """Write a :class:`PoseGraph` back out in ``.g2o`` format."""
    path = Path(path)
    out: List[str] = []
    if graph.space == "SE2":
        for nid in graph.pose_ids:
            p = graph.poses[graph._pose_index[nid]]
            out.append(f"VERTEX_SE2 {nid} " + " ".join(_fmt(v) for v in p))
        for pid in graph.point_ids:
            q = graph.points[graph._point_index[pid]]
            out.append(f"VERTEX_XY {pid} " + " ".join(_fmt(v) for v in q))
        for k in range(graph.num_edges):
            z = graph.edge_z[k]
            info = _upper_from_info(graph.edge_info[k])
            out.append(
                f"EDGE_SE2 {int(graph.edge_i[k])} {int(graph.edge_j[k])} "
                + " ".join(_fmt(v) for v in z)
                + " "
                + " ".join(_fmt(v) for v in info)
            )
        for k in range(graph.num_landmark_edges):
            info = _upper_from_info(graph.lm_info[k])
            out.append(
                f"EDGE_SE2_XY {int(graph.lm_pose[k])} {int(graph.lm_point[k])} "
                + " ".join(_fmt(v) for v in graph.lm_z[k])
                + " "
                + " ".join(_fmt(v) for v in info)
            )
    else:
        for nid in graph.pose_ids:
            p = graph.poses[graph._pose_index[nid]]
            out.append(f"VERTEX_SE3:QUAT {nid} " + " ".join(_fmt(v) for v in p))
        for pid in graph.point_ids:
            q = graph.points[graph._point_index[pid]]
            out.append(f"VERTEX_TRACKXYZ {pid} " + " ".join(_fmt(v) for v in q))
        for k in range(graph.num_edges):
            info = _upper_from_info(graph.edge_info[k])
            out.append(
                f"EDGE_SE3:QUAT {int(graph.edge_i[k])} {int(graph.edge_j[k])} "
                + " ".join(_fmt(v) for v in graph.edge_z[k])
                + " "
                + " ".join(_fmt(v) for v in info)
            )
        for k in range(graph.num_landmark_edges):
            info = _upper_from_info(graph.lm_info[k])
            out.append(
                f"EDGE_SE3_TRACKXYZ {int(graph.lm_pose[k])} {int(graph.lm_point[k])} 0 "
                + " ".join(_fmt(v) for v in graph.lm_z[k])
                + " "
                + " ".join(_fmt(v) for v in info)
            )
    for nid in graph.fixed:
        out.append(f"FIX {nid}")
    out.extend(graph.trailing_lines)
    path.write_text("\n".join(out) + "\n")
