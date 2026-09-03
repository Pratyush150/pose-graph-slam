"""Dataset registry, and end-to-end runs on the real benchmarks when present.

Every test that needs a downloaded file is skipped, never failed, when the file
is missing: the datasets are fetched on demand and are not in the repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DATA_DIR, requires_dataset
from posegraph import analysis, datasets, linalg, se3
from posegraph.solver import SolverOptions, optimize

#: The pure-NumPy Cholesky is correct but runs its elimination loop in Python,
#: which makes a 2500-pose SE(3) graph take minutes rather than seconds. The
#: large-dataset tests are therefore skipped without SciPy; the fallback itself
#: is covered by tests/test_linalg_backends.py and test_solver.py.
needs_sparse_backend = pytest.mark.skipif(
    not linalg.HAVE_SCIPY,
    reason="large benchmark solves need the SciPy sparse backend to run quickly",
)


def test_registry_entries_are_well_formed():
    assert len(datasets.REGISTRY) >= 12
    for name, spec in datasets.REGISTRY.items():
        assert spec.name == name
        assert spec.space in ("SE2", "SE3")
        assert spec.url.startswith("https://")
        assert len(spec.sha256) == 64
        assert int(spec.size_bytes) > 0
        assert spec.licence
        assert spec.origin
        if spec.ground_truth:
            assert spec.ground_truth in datasets.GROUND_TRUTH


def test_ground_truth_entries_are_well_formed():
    for name, spec in datasets.GROUND_TRUTH.items():
        assert spec.name == name
        assert len(spec.sha256) == 64
        assert spec.fmt in ("xytheta", "edge3_chain")


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        datasets.dataset_path("does-not-exist")


def test_missing_file_gives_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_datasets"):
        datasets.load("intel", data_dir=tmp_path)


@requires_dataset("intel")
def test_checksum_of_a_downloaded_file_matches_the_registry():
    path = datasets.dataset_path("intel", DATA_DIR)
    assert datasets.sha256_of(path) == datasets.REGISTRY["intel"].sha256


@requires_dataset("smallGrid3D")
def test_se3_benchmark_optimises():
    g = datasets.load("smallGrid3D", DATA_DIR)
    g.fix_pose(g.pose_ids[0])
    before = g.chi2()
    result = optimize(g, SolverOptions(method="lm", max_iterations=100))
    assert result.final_chi2 < before
    assert result.final_chi2 < 0.5 * before


@requires_dataset("intel")
@needs_sparse_backend
def test_intel_optimises_and_reduces_fill():
    g = datasets.load("intel", DATA_DIR)
    g.fix_pose(g.pose_ids[0])
    result = optimize(g, SolverOptions(method="lm", max_iterations=100))
    assert result.final_chi2 < result.initial_chi2
    assert result.ordering is not None
    assert result.ordering.chosen_nnz < result.ordering.natural_nnz


@requires_dataset("sphere2500_gt")
def test_sphere_ground_truth_is_internally_consistent():
    """The loop-closure records in the truth file must agree with the chained odometry.

    That is what makes it usable as ground truth rather than just another
    noisy trajectory.
    """
    poses = datasets.load_ground_truth("sphere2500_gt", DATA_DIR)
    assert poses.shape == (2500, 7)
    path = datasets.dataset_path("sphere2500_gt", DATA_DIR)
    worst = 0.0
    checked = 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "EDGE3":
            continue
        i, j = int(parts[1]), int(parts[2])
        if j == i + 1:
            continue
        predicted = se3.between(poses[i], poses[j])
        v = np.array([float(x) for x in parts[3:6]])
        worst = max(worst, float(np.abs(predicted[:3] - v).max()))
        checked += 1
    assert checked > 1000
    assert worst < 1e-3, f"loop closures disagree with the chain by {worst}"


@requires_dataset("manhattan")
@requires_dataset("manhattan_gt")
@needs_sparse_backend
def test_manhattan_ate_improves_against_real_ground_truth():
    g = datasets.load("manhattan", DATA_DIR)
    g.fix_pose(g.pose_ids[0])
    gt = datasets.load_ground_truth("manhattan_gt", DATA_DIR)
    assert gt.shape[0] == g.num_poses
    before = analysis.absolute_trajectory_error(g.poses, gt, "SE2")
    optimize(g, SolverOptions(method="lm", max_iterations=100))
    after = analysis.absolute_trajectory_error(g.poses, gt, "SE2")
    assert after.rmse < before.rmse


@requires_dataset("sphere2500")
@requires_dataset("sphere2500_gt")
@needs_sparse_backend
def test_sphere2500_ate_improves_against_real_ground_truth():
    g = datasets.load("sphere2500", DATA_DIR)
    g.fix_pose(g.pose_ids[0])
    gt = datasets.load_ground_truth("sphere2500_gt", DATA_DIR)
    before = analysis.absolute_trajectory_error(g.poses, gt, "SE3")
    optimize(g, SolverOptions(method="lm", max_iterations=100))
    after = analysis.absolute_trajectory_error(g.poses, gt, "SE3")
    assert after.rmse < 0.1 * before.rmse
