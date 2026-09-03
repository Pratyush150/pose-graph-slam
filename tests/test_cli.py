"""The command line surface, exercised end to end on temporary files."""

from __future__ import annotations

import pytest

from posegraph import cli, frontend_stub
from posegraph.graph import read_g2o, write_g2o


@pytest.fixture
def graph_file(tmp_path):
    sim = frontend_stub.simulate_se2(num_poses=80, turn_every=10, min_index_gap=25, seed=8)
    path = tmp_path / "sim.g2o"
    write_g2o(sim.graph, path)
    return path


def test_info_command(graph_file, capsys):
    assert cli.main(["info", str(graph_file)]) == 0
    out = capsys.readouterr().out
    assert "PoseGraph(SE2)" in out
    assert "chi2/dof" in out
    assert "connected components: 1" in out


def test_optimize_command_writes_output(graph_file, tmp_path, capsys):
    out_path = tmp_path / "opt.g2o"
    rc = cli.main(["optimize", str(graph_file), "-o", str(out_path), "--iterations", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "converged" in out
    assert out_path.exists()
    before = read_g2o(graph_file)
    after = read_g2o(out_path)
    assert after.chi2() < before.chi2()


def test_optimize_command_with_a_robust_kernel(graph_file, capsys):
    assert cli.main(["optimize", str(graph_file), "--kernel", "huber", "--iterations", "200"]) == 0
    assert "chi2" in capsys.readouterr().out


def test_demo_command(capsys):
    assert cli.main(["demo", "--poses", "80", "--iterations", "30"]) == 0
    out = capsys.readouterr().out
    assert "before" in out and "after" in out
    assert "not a benchmark result" in out


def test_datasets_command(capsys):
    assert cli.main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "sphere2500" in out and "manhattan" in out


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["nope"])
