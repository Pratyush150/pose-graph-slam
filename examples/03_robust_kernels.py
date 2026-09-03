#!/usr/bin/env python3
"""What one wrong loop closure does, and what a robust kernel does about it.

Runs on the synthetic circuit by default so it needs no downloads; pass a
registry dataset name to run the same experiment on a real benchmark.

    python3 examples/03_robust_kernels.py
    python3 examples/03_robust_kernels.py --dataset manhattan --false 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posegraph import analysis, datasets, frontend_stub  # noqa: E402
from posegraph.solver import SolverOptions, optimize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None, help="registry name; default is the synthetic stub")
    ap.add_argument("--false", type=int, default=8, help="number of false loop closures")
    ap.add_argument("--poses", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    ground_truth = None
    if args.dataset:
        base = datasets.load(args.dataset)
        base.fix_pose(base.pose_ids[0])
        gt_name = datasets.REGISTRY[args.dataset].ground_truth
        if gt_name:
            try:
                ground_truth = datasets.load_ground_truth(gt_name)
            except FileNotFoundError:
                pass
        label = args.dataset
    else:
        sim = frontend_stub.simulate_se2(
            num_poses=args.poses, turn_every=25, min_index_gap=40, seed=args.seed
        )
        base = sim.graph
        ground_truth = sim.ground_truth
        label = "synthetic circuit"

    corrupted, injected = frontend_stub.inject_false_loop_closures(
        base, args.false, seed=args.seed, min_index_gap=40, translation_scale=15.0
    )
    print(f"{label}: {base.num_poses} poses, {base.num_edges} edges")
    print(f"injected {len(injected)} false loop closures\n")

    panels = []
    print(f"{'kernel':<16}{'final chi2':>14}{'iters':>7}{'ATE rmse':>12}"
          f"{'median chi2 (fake)':>21}{'median chi2 (real)':>21}")
    for name in (None, "huber", "cauchy", "geman_mcclure", "dcs"):
        g = corrupted.copy()
        res = optimize(g, SolverOptions(method="lm", max_iterations=150, kernel=name))
        chi = g.edge_chi2()
        clean = np.setdiff1d(np.arange(g.num_edges), injected)
        ate = (
            analysis.absolute_trajectory_error(g.poses, ground_truth, g.space).rmse
            if ground_truth is not None and ground_truth.shape[0] == g.num_poses
            else float("nan")
        )
        print(
            f"{str(name or 'none'):<16}{res.final_chi2:>14.4g}{res.iterations:>7}"
            f"{ate:>12.4f}{np.median(chi[injected]):>21.4g}{np.median(chi[clean]):>21.4g}"
        )
        panels.append((str(name or "no kernel"), g.poses.copy()))

    print(
        "\nRead the last two columns: a working kernel leaves the fabricated edges with a\n"
        "huge residual (it refused to bend the map to satisfy them) while the genuine\n"
        "edges stay near their noise floor. Without a kernel both columns move together,\n"
        "because the map was warped until the lies looked plausible."
    )

    if args.plot:
        from posegraph import plotting

        out = plotting.plot_kernel_comparison(
            panels, base.space, args.plot, title=f"{label}: {len(injected)} false loop closures",
            ground_truth=ground_truth if ground_truth is not None else None,
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
