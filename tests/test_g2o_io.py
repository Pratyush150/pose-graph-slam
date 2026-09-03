"""``.g2o`` parsing: round-trip fidelity and the awkward real-world cases."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DATA_DIR, requires_dataset
from posegraph.graph import read_g2o, write_g2o

SE2_TEXT = """VERTEX_SE2 0 0.0 0.0 0.0
VERTEX_SE2 1 1.0 0.5 0.25
VERTEX_SE2 2 2.0 1.0 0.5
EDGE_SE2 0 1 1.0 0.5 0.25 44.0 1.0 2.0 45.0 3.0 900.0
EDGE_SE2 1 2 1.0 0.5 0.25 44.0 1.0 2.0 45.0 3.0 900.0
EDGE_SE2 0 2 2.0 1.0 0.5 10.0 0.0 0.0 10.0 0.0 100.0
FIX 0
"""

SE3_TEXT = """VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1
VERTEX_SE3:QUAT 1 1 0 0 0 0 0.2474 0.9689
EDGE_SE3:QUAT 0 1 1 0 0 0 0 0.2474 0.9689 """ + " ".join(["1.0"] * 21) + "\nFIX 0\n"


def test_se2_round_trip_is_lossless(tmp_path):
    src = tmp_path / "in.g2o"
    src.write_text(SE2_TEXT)
    g = read_g2o(src)
    assert g.space == "SE2"
    assert g.num_poses == 3
    assert g.num_edges == 3
    assert g.fixed == [0]
    out = tmp_path / "out.g2o"
    write_g2o(g, out)
    h = read_g2o(out)
    assert h.pose_ids == g.pose_ids
    assert np.allclose(h.poses, g.poses, atol=0.0)
    assert np.array_equal(h.edge_i, g.edge_i)
    assert np.array_equal(h.edge_j, g.edge_j)
    assert np.allclose(h.edge_z, g.edge_z, atol=0.0)
    assert np.allclose(h.edge_info, g.edge_info, atol=0.0)
    assert h.fixed == g.fixed


def test_information_matrix_is_symmetric_and_correctly_placed(tmp_path):
    src = tmp_path / "in.g2o"
    src.write_text(SE2_TEXT)
    g = read_g2o(src)
    M = g.edge_info[0]
    assert np.allclose(M, M.T)
    assert M[0, 0] == 44.0 and M[0, 1] == 1.0 and M[0, 2] == 2.0
    assert M[1, 1] == 45.0 and M[1, 2] == 3.0 and M[2, 2] == 900.0


def test_se3_round_trip_is_lossless(tmp_path):
    src = tmp_path / "in3.g2o"
    src.write_text(SE3_TEXT)
    g = read_g2o(src)
    assert g.space == "SE3"
    assert g.num_poses == 2 and g.num_edges == 1
    assert np.isclose(np.linalg.norm(g.poses[1, 3:]), 1.0)
    out = tmp_path / "out3.g2o"
    write_g2o(g, out)
    h = read_g2o(out)
    assert np.allclose(h.poses, g.poses, atol=0.0)
    assert np.allclose(h.edge_z, g.edge_z, atol=0.0)
    assert np.allclose(h.edge_info, g.edge_info, atol=0.0)


def test_optimised_values_survive_a_write_read_cycle(tmp_path):
    from posegraph import frontend_stub
    from posegraph.solver import optimize, SolverOptions

    sim = frontend_stub.simulate_se2(num_poses=40, seed=7)
    optimize(sim.graph, SolverOptions(max_iterations=30))
    path = tmp_path / "opt.g2o"
    write_g2o(sim.graph, path)
    back = read_g2o(path)
    assert np.allclose(back.poses, sim.graph.poses, atol=0.0), "full float precision on write"
    assert np.isclose(back.chi2(), sim.graph.chi2(), rtol=1e-12)


def test_unknown_lines_are_preserved(tmp_path):
    src = tmp_path / "extra.g2o"
    src.write_text(SE2_TEXT + "PARAMS_SE3OFFSET 0 0 0 0 0 0 0 1\n# a comment\n")
    g = read_g2o(src)
    assert any("PARAMS_SE3OFFSET" in ln for ln in g.trailing_lines)
    assert any(ln.startswith("#") for ln in g.trailing_lines)
    out = tmp_path / "out.g2o"
    write_g2o(g, out)
    assert "PARAMS_SE3OFFSET" in out.read_text()


def test_edges_only_file_gets_a_spanning_tree_guess(tmp_path):
    src = tmp_path / "edges.g2o"
    src.write_text(
        "EDGE_SE2 0 1 1.0 0.0 0.0 1 0 0 1 0 1\n"
        "EDGE_SE2 1 2 1.0 0.0 0.0 1 0 0 1 0 1\n"
        "EDGE_SE2 2 3 1.0 0.0 0.0 1 0 0 1 0 1\n"
    )
    g = read_g2o(src)
    assert g.num_poses == 4
    assert np.allclose(g.pose(3), np.array([3.0, 0.0, 0.0]), atol=1e-12)
    assert g.chi2() < 1e-20


def test_landmark_records_round_trip(tmp_path):
    src = tmp_path / "lm.g2o"
    src.write_text(
        "VERTEX_SE2 0 0 0 0\nVERTEX_XY 100 1.5 2.5\n"
        "EDGE_SE2_XY 0 100 1.5 2.5 10.0 0.0 10.0\n"
    )
    g = read_g2o(src)
    assert g.num_points == 1 and g.num_landmark_edges == 1
    assert np.allclose(g.landmark_errors(), 0.0, atol=1e-12)
    out = tmp_path / "lm_out.g2o"
    write_g2o(g, out)
    h = read_g2o(out)
    assert np.allclose(h.points, g.points, atol=0.0)
    assert np.allclose(h.lm_z, g.lm_z, atol=0.0)


def test_malformed_line_raises(tmp_path):
    src = tmp_path / "bad.g2o"
    src.write_text("VERTEX_SE2 0 0.0 notanumber 0.0\n")
    with pytest.raises(ValueError):
        read_g2o(src)


def test_empty_file_raises(tmp_path):
    src = tmp_path / "empty.g2o"
    src.write_text("\n\n")
    with pytest.raises(ValueError):
        read_g2o(src)


@requires_dataset("intel")
def test_real_intel_file_parses():
    g = read_g2o(DATA_DIR / "intel.g2o")
    assert g.space == "SE2"
    assert g.num_poses == 1728
    assert g.num_edges == 2512
    assert np.all(np.linalg.eigvalsh(g.edge_info) > 0), "information matrices must be SPD"


@requires_dataset("sphere2500")
def test_real_sphere_file_parses():
    g = read_g2o(DATA_DIR / "sphere2500.g2o")
    assert g.space == "SE3"
    assert g.num_poses == 2500
    assert g.num_edges == 4949
    assert np.allclose(np.linalg.norm(g.poses[:, 3:], axis=1), 1.0, atol=1e-9)


@requires_dataset("manhattan")
def test_manhattan_ships_edges_only():
    g = read_g2o(DATA_DIR / "manhattan.g2o")
    assert g.num_poses == 3500
    assert g.num_edges == 5453
    assert len(g.connected_components()) == 1
