"""Figure generation. Skipped whole if matplotlib is not installed."""

from __future__ import annotations

import pytest

from posegraph import frontend_stub, linalg
from posegraph.solver import SolverOptions, Problem, optimize

pytest.importorskip("matplotlib", reason="matplotlib is an optional dependency")

from posegraph import plotting  # noqa: E402


@pytest.fixture(scope="module")
def solved():
    sim = frontend_stub.simulate_se2(num_poses=120, turn_every=10, min_index_gap=25, seed=6)
    before = sim.graph.poses.copy()
    result = optimize(sim.graph, SolverOptions(method="lm", max_iterations=50))
    return sim, before, result


def test_before_after_figure(tmp_path, solved):
    sim, before, _result = solved
    out = plotting.plot_before_after(
        before, sim.graph.poses, "SE2", tmp_path / "ba.png", ground_truth=sim.ground_truth
    )
    assert out.exists() and out.stat().st_size > 1000


def test_trajectory_figure(tmp_path, solved):
    sim, _before, _result = solved
    out = plotting.plot_trajectory(sim.graph.poses, "SE2", tmp_path / "t.png")
    assert out.exists() and out.stat().st_size > 500


def test_convergence_figure(tmp_path, solved):
    _sim, _before, result = solved
    out = plotting.plot_convergence(
        {"lm": result.chi2_history},
        tmp_path / "c.png",
        initial={"lm": result.initial_chi2},
    )
    assert out.exists() and out.stat().st_size > 500


def test_kernel_comparison_figure(tmp_path, solved):
    sim, before, _result = solved
    out = plotting.plot_kernel_comparison(
        [("no kernel", before), ("dcs", sim.graph.poses)], "SE2", tmp_path / "k.png"
    )
    assert out.exists() and out.stat().st_size > 1000


def test_sparsity_figure(tmp_path, solved):
    sim, _before, _result = solved
    problem = Problem(sim.graph)
    report = linalg.fill_report(problem.block_adj)
    out = plotting.plot_sparsity(
        problem.block_adj,
        report.perm,
        tmp_path / "s.png",
        fill=(report.natural_nnz, report.chosen_nnz),
    )
    assert out.exists() and out.stat().st_size > 1000


def test_residual_histogram(tmp_path, solved):
    sim, _before, _result = solved
    bad, idx = frontend_stub.inject_false_loop_closures(sim.graph, 4, seed=1, min_index_gap=25)
    out = plotting.plot_residual_histogram(bad.edge_chi2(), tmp_path / "h.png", highlight=idx)
    assert out.exists() and out.stat().st_size > 500


def test_fill_pattern_figure(tmp_path, solved):
    sim, _before, _result = solved
    problem = Problem(sim.graph)
    report = linalg.fill_report(problem.block_adj)
    out = plotting.plot_fill_pattern(problem.block_adj, report.perm, tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 1000
