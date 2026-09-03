"""Robust kernels: correct derivatives, and a demonstrated effect on an outlier."""

from __future__ import annotations

import numpy as np
import pytest

from posegraph import frontend_stub, robust
from posegraph.analysis import rank_outliers
from posegraph.solver import SolverOptions, optimize

ALL = ["huber", "cauchy", "geman_mcclure", "dcs"]


@pytest.mark.parametrize("name", ALL + ["trivial"])
def test_kernel_is_quadratic_for_small_residuals(name):
    """Every kernel must leave inliers alone, or it biases a clean graph."""
    k = robust.make_kernel(name, delta=1.0)
    s = np.array([1e-9, 1e-7, 1e-5])
    rho, w = k.evaluate(s)
    assert np.allclose(rho, s, rtol=1e-3)
    assert np.allclose(w, 1.0, rtol=1e-3)


@pytest.mark.parametrize("name", ALL)
def test_weight_is_the_derivative_of_the_cost(name):
    """``w(s) = drho/ds`` checked against central differences."""
    k = robust.make_kernel(name, delta=1.3)
    for s0 in (0.05, 0.5, 1.0, 2.0, 10.0, 100.0):
        h = 1e-6 * max(s0, 1.0)
        num = (k.cost(np.array([s0 + h]))[0] - k.cost(np.array([s0 - h]))[0]) / (2.0 * h)
        assert abs(num - k.weight(np.array([s0]))[0]) < 1e-5


@pytest.mark.parametrize("name", ALL)
def test_weights_decrease_with_residual(name):
    k = robust.make_kernel(name, delta=1.0)
    s = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    w = k.weight(s)
    assert np.all(np.diff(w) <= 1e-12)
    assert w[-1] < 0.2, "a gross outlier must be strongly down-weighted"


@pytest.mark.parametrize("name", ALL)
def test_cost_grows_more_slowly_than_squared(name):
    k = robust.make_kernel(name, delta=1.0)
    s = np.array([100.0, 400.0])
    rho = k.cost(s)
    assert rho[1] < s[1]
    assert rho[1] / rho[0] < 4.0, "sub-quadratic growth is the whole point"


def test_huber_matches_its_closed_form():
    k = robust.HuberKernel(delta=2.0)
    rho, w = k.evaluate(np.array([1.0, 16.0]))
    assert np.isclose(rho[0], 1.0) and np.isclose(w[0], 1.0)
    assert np.isclose(rho[1], 2.0 * 2.0 * 4.0 - 4.0)
    assert np.isclose(w[1], 2.0 / 4.0)


def test_cauchy_matches_its_closed_form():
    k = robust.CauchyKernel(delta=1.5)
    c2 = 2.25
    s = np.array([3.0])
    assert np.isclose(k.cost(s)[0], c2 * np.log(1.0 + 3.0 / c2))
    assert np.isclose(k.weight(s)[0], 1.0 / (1.0 + 3.0 / c2))


def test_dcs_saturates_at_full_weight_below_phi():
    k = robust.DCSKernel(delta=2.0)  # Phi = 4
    assert np.isclose(k.weight(np.array([1.0]))[0], 1.0)
    assert np.isclose(k.weight(np.array([4.0]))[0], 1.0)
    assert k.weight(np.array([40.0]))[0] < 0.1


def test_make_kernel_accepts_names_objects_and_none():
    assert isinstance(robust.make_kernel(None), robust.TrivialKernel)
    assert isinstance(robust.make_kernel("HUBER"), robust.HuberKernel)
    k = robust.CauchyKernel(delta=3.0)
    assert robust.make_kernel(k) is k
    with pytest.raises(ValueError):
        robust.make_kernel("bogus")
    with pytest.raises(ValueError):
        robust.HuberKernel(delta=0.0)


def _corrupted_graph(n_false=6, seed=3):
    sim = frontend_stub.simulate_se2(num_poses=120, turn_every=10, min_index_gap=25, seed=seed)
    bad, injected = frontend_stub.inject_false_loop_closures(
        sim.graph, n_false, seed=seed, min_index_gap=25, translation_scale=8.0
    )
    return sim, bad, injected


def test_robust_kernel_down_weights_the_injected_outliers():
    sim, bad, injected = _corrupted_graph()
    g = bad.copy()
    optimize(g, SolverOptions(method="lm", max_iterations=80, kernel="dcs", kernel_delta=1.0))
    chi = g.edge_chi2()
    clean = np.setdiff1d(np.arange(g.num_edges), injected)
    kernel = robust.make_kernel("dcs", 1.0)
    w = kernel.weight(chi)
    assert w[injected].max() < 0.5, "injected edges must be suppressed"
    assert np.median(w[clean]) > 0.9, "genuine edges must keep their weight"


def test_outlier_ranking_finds_the_injected_edges():
    sim, bad, injected = _corrupted_graph(n_false=5)
    g = bad.copy()
    optimize(g, SolverOptions(method="lm", max_iterations=80, kernel="dcs"))
    worst = [r.index for r in rank_outliers(g, top=len(injected))]
    assert set(worst) == set(injected.tolist())


def test_map_survives_outliers_with_a_kernel_and_not_without():
    """The headline claim, asserted rather than asserted-about."""
    from posegraph.analysis import absolute_trajectory_error

    sim, bad, injected = _corrupted_graph(n_false=8, seed=9)

    plain = bad.copy()
    optimize(plain, SolverOptions(method="lm", max_iterations=100))
    ate_plain = absolute_trajectory_error(plain.poses, sim.ground_truth, "SE2")

    robustified = bad.copy()
    optimize(robustified, SolverOptions(method="lm", max_iterations=100, kernel="dcs"))
    ate_robust = absolute_trajectory_error(robustified.poses, sim.ground_truth, "SE2")

    assert ate_robust.rmse < ate_plain.rmse
    assert ate_robust.rmse < 0.5 * ate_plain.rmse


def test_default_delta_comes_from_the_chi_squared_table():
    """delta = 1 treats an ordinary residual as an outlier; this picks it properly."""
    assert np.isclose(robust.default_delta(3), np.sqrt(7.815), atol=1e-9)
    assert np.isclose(robust.default_delta(6), np.sqrt(12.592), atol=1e-9)
    assert robust.default_delta(6, "0.99") > robust.default_delta(6, "0.95")
    with pytest.raises(ValueError):
        robust.default_delta(3, "0.5")
    with pytest.raises(ValueError):
        robust.default_delta(9)


def test_default_delta_leaves_typical_inliers_untouched():
    """A residual at the median of a chi2(3) distribution must keep full weight."""
    k = robust.make_kernel("huber", robust.default_delta(3))
    assert np.isclose(k.weight(np.array([2.366]))[0], 1.0), "chi2(3) median"
    assert k.weight(np.array([100.0]))[0] < 0.3
