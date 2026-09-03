#!/usr/bin/env python3
"""Smallest end-to-end run: simulate a drifting loop, then close it.

No downloads, no hardware, no external services. Roughly one second.

    python3 examples/01_quickstart_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posegraph import analysis, frontend_stub  # noqa: E402
from posegraph.solver import SolverOptions, optimize  # noqa: E402


def main() -> int:
    sim = frontend_stub.simulate_se2(
        num_poses=400, turn_every=25, min_index_gap=40, seed=1
    )
    g = sim.graph
    print(g.summary())
    print(f"loop closures found by the stub front end: {sim.num_loop_closures}")

    before = analysis.absolute_trajectory_error(g.poses, sim.ground_truth, "SE2")
    print("\nbefore optimisation")
    print("  " + before.describe())
    print("  " + analysis.chi2_report(g).describe())

    result = optimize(g, SolverOptions(method="lm", max_iterations=100))
    print("\n" + result.summary())
    print("  stop reason: " + result.message)
    if result.ordering:
        print("  " + result.ordering.describe())

    after = analysis.absolute_trajectory_error(g.poses, sim.ground_truth, "SE2")
    rpe = analysis.relative_pose_error(g.poses, sim.ground_truth, "SE2", delta=1)
    print("\nafter optimisation")
    print("  " + after.describe())
    print("  " + rpe.describe())
    print("  " + analysis.chi2_report(g).describe())
    print(f"\nATE improved by {100.0 * (1.0 - after.rmse / before.rmse):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
