"""Registry of the public benchmark pose graphs, and loaders for their truth files.

None of these files are redistributed with this package. The mirrors they are
fetched from carry licences (LGPL-3.0, GPL-3.0, BSD) that are not compatible
with shipping the data inside an MIT repository, so ``tools/fetch_datasets.py``
downloads them on demand into ``data/`` and verifies the SHA-256 of every file
against the values recorded here. The checksums were taken from the files this
package was actually benchmarked against, so a mirror that silently changes is
caught rather than quietly producing different numbers.

Provenance of the datasets themselves is in ``docs/datasets.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import se3
from .graph import PoseGraph, read_g2o

__all__ = [
    "DatasetSpec",
    "REGISTRY",
    "GROUND_TRUTH",
    "default_data_dir",
    "dataset_path",
    "available",
    "sha256_of",
    "load",
    "load_ground_truth",
]

_SESYNC = "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/"
_GTSAM = "https://raw.githubusercontent.com/borglab/gtsam/develop/examples/Data/"
_VERTIGO = (
    "https://raw.githubusercontent.com/OpenSLAM-org/openslam_vertigo/master/datasets/"
)


@dataclass(frozen=True)
class DatasetSpec:
    """One downloadable benchmark file."""

    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    space: str
    origin: str
    licence: str
    ground_truth: Optional[str] = None
    note: str = ""


REGISTRY: Dict[str, DatasetSpec] = {
    d.name: d
    for d in [
        DatasetSpec(
            "intel",
            "intel.g2o",
            _SESYNC + "intel.g2o",
            "3e0724c048e0ba524be9dd268a8b78e19a2497043143584cbb61310638b15c4b",
            307525,
            "SE2",
            "Intel Research Lab, Seattle (Dirk Haehnel); g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="The classic indoor 2D loop-closure test.",
        ),
        DatasetSpec(
            "MIT",
            "MIT.g2o",
            _SESYNC + "MIT.g2o",
            "e5922be0d0689c7a5bc04c58adf3a8e697e240bdd7691cc4218470eaf92956eb",
            121424,
            "SE2",
            "MIT Killian Court (Mike Bosse); g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Very sparse loop closures and long open corridors.",
        ),
        DatasetSpec(
            "CSAIL",
            "CSAIL.g2o",
            _SESYNC + "CSAIL.g2o",
            "66d99ac857a9849d814d214a9ebd0d4876d5d40f0a37be9330c1ff6e6e9daaa6",
            127618,
            "SE2",
            "MIT CSAIL building; g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Indoor 2D office loop.",
        ),
        DatasetSpec(
            "manhattan",
            "manhattan.g2o",
            _SESYNC + "manhattan.g2o",
            "6ae8d30971720c1af24a00c4b2dd5c5ddafbbbe488bfc771145c47decbffb248",
            594668,
            "SE2",
            "Manhattan3500 / M3500 (Ed Olson); g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            ground_truth="manhattan_gt",
            note="Ships edges only, so the initial guess must be chained from odometry.",
        ),
        DatasetSpec(
            "city10000",
            "city10000.g2o",
            _SESYNC + "city10000.g2o",
            "df5988994339e990be198a36e7f640e31a5a1b26df3ed400363fafc49d5ca630",
            1750622,
            "SE2",
            "City10000 simulated urban sequence; g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            ground_truth="city10000_gt",
            note="Simulated city grid, the densest 2D loop structure here.",
        ),
        DatasetSpec(
            "ais2klinik",
            "ais2klinik.g2o",
            _SESYNC + "ais2klinik.g2o",
            "e388ebd8523987c1004082d7eaeb5e6fb640b773d05ccebeef1ce1e8df64e1e7",
            2236921,
            "SE2",
            "AIS2 Klinik, University of Freiburg; g2o conversion via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="The largest 2D graph here; long corridors, few loops.",
        ),
        DatasetSpec(
            "sphere2500",
            "sphere2500.g2o",
            _SESYNC + "sphere2500.g2o",
            "104ab57593394f24351d9f692f3b923f8b98fff1eb638c64356cf5049e06cf3c",
            1094712,
            "SE3",
            "sphere2500 simulated 3D sphere (Michael Kaess); via SE-Sync",
            "LGPL-3.0 (mirror)",
            ground_truth="sphere2500_gt",
            note="The standard SE(3) stress test: large rotations everywhere.",
        ),
        DatasetSpec(
            "torus3D",
            "torus3D.g2o",
            _SESYNC + "torus3D.g2o",
            "1ef8831023b372188d33ca63ce0da079d51df89364df71f2bce1c5c7c69ed5ec",
            3262828,
            "SE3",
            "torus3D simulated 3D torus; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Closes in three dimensions at once; slow to converge.",
        ),
        DatasetSpec(
            "parking-garage",
            "parking-garage.g2o",
            _SESYNC + "parking-garage.g2o",
            "3ac0a31bfb601d7455d451e2546655cb5dececf51a7823f57c8a7e0fe1ca6527",
            1281113,
            "SE3",
            "Multi-level parking garage, real vehicle data; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Real vehicle data, many short-range constraints.",
        ),
        DatasetSpec(
            "cubicle",
            "cubicle.g2o",
            _SESYNC + "cubicle.g2o",
            "f7781d485383cec86d47d7650970132c36d6f3a1f4e5d62a49b7f8245c0a6465",
            2687081,
            "SE3",
            "Cubicle office environment; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Densest SE(3) loop-closure structure here.",
        ),
        DatasetSpec(
            "grid3D",
            "grid3D.g2o",
            _SESYNC + "grid3D.g2o",
            "f5dd7a17bf8b9c39890b14f9bb0c1e8366a7490f01714e43e3bc03f52cf414e7",
            7645820,
            "SE3",
            "Simulated 3D grid; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Simulated 3D grid.",
        ),
        DatasetSpec(
            "rim",
            "rim.g2o",
            _SESYNC + "rim.g2o",
            "9cc9b0d8e5f080c5a029605e6cffd304f3c74d404d13f6329d942c473591155b",
            4857469,
            "SE3",
            "RIM dataset; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Large 3D graph.",
        ),
        DatasetSpec(
            "smallGrid3D",
            "smallGrid3D.g2o",
            _SESYNC + "smallGrid3D.g2o",
            "9ea56c2ad1ebcc322560eb2f8d83cb3a60f99e2e2acc35e097b1162cdbafd649",
            101258,
            "SE3",
            "Small simulated 3D grid; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Fast smoke test for the SE(3) path.",
        ),
        DatasetSpec(
            "tinyGrid3D",
            "tinyGrid3D.g2o",
            _SESYNC + "tinyGrid3D.g2o",
            "c341eb0d09f7556b337be5a62b9354384885333a25fa718fd699fafb19620493",
            4106,
            "SE3",
            "Tiny simulated 3D grid; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Nine poses; small enough to inspect by hand.",
        ),
        DatasetSpec(
            "sphere_bignoise",
            "sphere_bignoise_vertex3.g2o",
            _SESYNC + "sphere_bignoise_vertex3.g2o",
            "484aa1999084d353d83725ba1d992cb709ad3a7e6c396155cc8e87a059c645db",
            2221712,
            "SE3",
            "sphere2500 with inflated measurement noise; via SE-Sync",
            "LGPL-3.0 (mirror)",
            note="Inflated measurement noise; a much harder initial guess than sphere2500.",
        ),
    ]
}


@dataclass(frozen=True)
class GroundTruthSpec:
    """A ground-truth file for one of the registry datasets."""

    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    fmt: str
    space: str
    origin: str
    licence: str


GROUND_TRUTH: Dict[str, GroundTruthSpec] = {
    g.name: g
    for g in [
        GroundTruthSpec(
            "manhattan_gt",
            "manhattanOlson3500_nodes_groundTruth.dat",
            _VERTIGO + "manhattan/groundTruth/manhattanOlson3500_nodes_groundTruth.dat",
            "8512b270849f2571ace2f5fac486d3ee9b686e8334c7c55a6f96c2025a413181",
            89448,
            "xytheta",
            "SE2",
            "Vertigo (Niko Suenderhauf), ground truth for Olson's M3500",
            "GPL-3.0",
        ),
        GroundTruthSpec(
            "city10000_gt",
            "ISAM2_GT_city10000.txt",
            _GTSAM + "ISAM2_GT_city10000.txt",
            "a300f29b446608ea1c0175b5a7463584b82200196ebf0ac24cf1291744f97eb2",
            216006,
            "xytheta",
            "SE2",
            "GTSAM example data",
            "BSD-style (see GTSAM LICENSE)",
        ),
        GroundTruthSpec(
            "sphere2500_gt",
            "sphere2500_groundtruth.txt",
            _GTSAM + "sphere2500_groundtruth.txt",
            "b9cfd29c951586bf9afc09bb8f88bf67b7436e6c988a3e208e126e7d77b4520a",
            633111,
            "edge3_chain",
            "SE3",
            "GTSAM example data (noise-free EDGE3 records)",
            "BSD-style (see GTSAM LICENSE)",
        ),
    ]
}


def default_data_dir() -> Path:
    """``<repo>/data``, resolved relative to this file."""
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def dataset_path(name: str, data_dir: Path | str | None = None) -> Path:
    """Where dataset ``name`` is expected to live on disk."""
    spec = REGISTRY.get(name) or GROUND_TRUTH.get(name)
    if spec is None:
        raise KeyError(f"unknown dataset {name!r}")
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    return root / spec.filename


def available(data_dir: Path | str | None = None) -> List[str]:
    """Names of registry datasets present on disk right now."""
    return [n for n in REGISTRY if dataset_path(n, data_dir).exists()]


def sha256_of(path: Path | str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load(name: str, data_dir: Path | str | None = None, verify: bool = False) -> PoseGraph:
    """Read a registry dataset from disk into a :class:`PoseGraph`."""
    spec = REGISTRY[name]
    path = dataset_path(name, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python3 tools/fetch_datasets.py {name}"
        )
    if verify:
        got = sha256_of(path)
        if got != spec.sha256:
            raise ValueError(f"{path}: sha256 mismatch (expected {spec.sha256}, got {got})")
    return read_g2o(path, space=spec.space)


def _rpy_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Z-Y-X (yaw-pitch-roll) Euler angles to a scalar-last quaternion."""
    zero = np.zeros_like(yaw)
    qz = np.atleast_2d(se3.so3_exp(np.stack([zero, zero, yaw], 1)))
    qy = np.atleast_2d(se3.so3_exp(np.stack([zero, pitch, zero], 1)))
    qx = np.atleast_2d(se3.so3_exp(np.stack([roll, zero, zero], 1)))
    return np.atleast_2d(se3.quat_multiply(se3.quat_multiply(qz, qy), qx))


def load_ground_truth(
    name: str, data_dir: Path | str | None = None, verify: bool = False
) -> np.ndarray:
    """Load a ground-truth trajectory as a pose array.

    Two formats are handled:

    ``xytheta``
        One ``x y theta`` line per pose.
    ``edge3_chain``
        Noise-free legacy ``EDGE3`` records. The sequential edges are chained
        from the identity to recover absolute poses; the loop-closure records
        in the same file are then a consistency check on that chain, which
        ``tests/test_datasets.py`` runs when the file is present.
    """
    spec = GROUND_TRUTH[name]
    path = dataset_path(name, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python3 tools/fetch_datasets.py {name}"
        )
    if verify:
        got = sha256_of(path)
        if got != spec.sha256:
            raise ValueError(f"{path}: sha256 mismatch (expected {spec.sha256}, got {got})")
    if spec.fmt == "xytheta":
        return np.loadtxt(path, dtype=float).reshape(-1, 3)
    if spec.fmt == "edge3_chain":
        seq: Dict[int, np.ndarray] = {}
        max_id = 0
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if not parts or parts[0] != "EDGE3":
                continue
            i, j = int(parts[1]), int(parts[2])
            max_id = max(max_id, i, j)
            if j != i + 1:
                continue
            v = np.array([float(x) for x in parts[3:9]])
            q = _rpy_to_quat(
                np.array([v[3]]), np.array([v[4]]), np.array([v[5]])
            ).reshape(4)
            seq[i] = np.concatenate([v[:3], q])
        n = max_id + 1
        poses = np.zeros((n, 7))
        poses[:, 6] = 1.0
        for k in range(1, n):
            if k - 1 not in seq:
                raise ValueError(f"{path}: sequential edge {k - 1}->{k} missing")
            poses[k] = se3.compose(poses[k - 1], seq[k - 1])
        return poses
    raise ValueError(f"unsupported ground-truth format {spec.fmt!r}")
