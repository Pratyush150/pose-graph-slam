#!/usr/bin/env python3
"""Optimise a real ``.g2o`` benchmark and write the result back out.

    python3 tools/fetch_datasets.py intel
    python3 examples/02_optimise_g2o.py intel
    python3 examples/02_optimise_g2o.py sphere2500 --plot /tmp/sphere.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posegraph import analysis, datasets  # noqa: E402
from posegraph.graph import read_g2o, write_g2o  # noqa: E402
from posegraph.solver import SolverOptions, optimize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", help="a registry name (e.g. intel) or a path to a .g2o file")
    ap.add_argument("--method", default="lm", choices=["lm", "gn", "dogleg"])
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--output", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    if args.dataset in datasets.REGISTRY:
        try:
            graph = datasets.load(args.dataset)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        gt_name = datasets.REGISTRY[args.dataset].ground_truth
    else:
        graph = read_g2o(args.dataset)
        gt_name = None

    graph.fix_pose(graph.pose_ids[0])
    print(graph.summary())
    before = graph.poses.copy()

    gt = None
    if gt_name:
        try:
            gt = datasets.load_ground_truth(gt_name)
        except FileNotFoundError:
            print(f"(ground truth {gt_name} not downloaded)")

    if gt is not None:
        ate = analysis.absolute_trajectory_error(before, gt, graph.space)
        print("  ATE before: " + ate.describe())

    result = optimize(
        graph,
        SolverOptions(method=args.method, max_iterations=args.iterations, kernel=args.kernel),
    )
    print(result.summary())
    print("  stop reason: " + result.message)
    if result.ordering:
        print("  " + result.ordering.describe())
    print("  " + analysis.chi2_report(graph).describe())
    if gt is not None:
        ate = analysis.absolute_trajectory_error(graph.poses, gt, graph.space)
        print("  ATE after:  " + ate.describe())

    print("\nworst remaining edges:")
    for r in analysis.rank_outliers(graph, top=5):
        print(f"  {r.i:>6} -> {r.j:<6} {r.tag:<10} chi2={r.chi2:12.4f}")

    if args.output:
        write_g2o(graph, args.output)
        print(f"\nwrote {args.output}")
    if args.plot:
        from posegraph import plotting

        out = plotting.plot_before_after(
            before, graph.poses, graph.space, args.plot, title=args.dataset, ground_truth=gt
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
