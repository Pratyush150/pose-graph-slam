"""Command line front end: ``posegraph optimize|info|demo``.

Installed as the ``posegraph`` entry point, and runnable in place with
``python3 -m posegraph.cli``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from . import analysis, datasets, frontend_stub
from .graph import read_g2o, write_g2o
from .solver import SolverOptions, optimize

__all__ = ["main"]


def _add_solver_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--method", default="lm", choices=["lm", "gn", "dogleg"])
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument(
        "--kernel",
        default=None,
        choices=["huber", "cauchy", "geman_mcclure", "dcs", "trivial"],
        help="robust kernel (default: none)",
    )
    ap.add_argument(
        "--kernel-delta",
        type=float,
        default=None,
        help="kernel width in units of sqrt(chi2); "
        "default is the chi-squared 95%% critical value for the problem dimension",
    )
    ap.add_argument("--ordering", default="auto", choices=["auto", "rcm", "md", "none"])
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "scipy_splu", "numpy_sparse", "dense"],
    )
    ap.add_argument("--verbose", action="store_true")


def _options(args) -> SolverOptions:
    return SolverOptions(
        method=args.method,
        max_iterations=args.iterations,
        kernel=args.kernel,
        kernel_delta=args.kernel_delta,
        ordering=args.ordering,
        backend=args.backend,
        verbose=args.verbose,
    )


def cmd_info(args) -> int:
    g = read_g2o(args.input)
    print(g.summary())
    rep = analysis.chi2_report(g)
    print(rep.describe())
    comps = g.connected_components()
    print(f"connected components: {len(comps)} (largest {max(len(c) for c in comps)} poses)")
    print(f"loop-closure edges: {len(g.loop_edges())}")
    worst = analysis.rank_outliers(g, top=args.top)
    if worst:
        print(f"\nworst {len(worst)} edges by chi2:")
        for r in worst:
            print(
                f"  {r.i:>6} -> {r.j:<6} {r.tag:<10} "
                f"chi2={r.chi2:12.4f}  sigma={r.mahalanobis:8.3f}"
            )
    return 0


def cmd_optimize(args) -> int:
    g = read_g2o(args.input)
    if args.fix is not None:
        g.fix_pose(args.fix)
    elif not g.fixed:
        g.fix_pose(g.pose_ids[0])
    before = g.poses.copy()
    print(g.summary())
    result = optimize(g, _options(args))
    print(result.summary())
    print(f"stop reason: {result.message}")
    if result.ordering is not None:
        print(result.ordering.describe())
    print(analysis.chi2_report(g).describe())
    if args.output:
        write_g2o(g, args.output)
        print(f"wrote {args.output}")
    if args.plot:
        from . import plotting

        out = plotting.plot_before_after(
            before, g.poses, g.space, args.plot, title=Path(args.input).name
        )
        print(f"wrote {out}")
    return 0 if result.converged else 1


def cmd_demo(args) -> int:
    print("Synthetic demo. This is the stub front end, not a benchmark result.")
    sim = (
        frontend_stub.simulate_se3(num_poses=args.poses, seed=args.seed)
        if args.se3
        else frontend_stub.simulate_se2(num_poses=args.poses, seed=args.seed)
    )
    g = sim.graph
    print(g.summary() + f"  loop closures={sim.num_loop_closures}")
    ate_before = analysis.absolute_trajectory_error(g.poses, sim.ground_truth, g.space)
    result = optimize(g, _options(args))
    ate_after = analysis.absolute_trajectory_error(g.poses, sim.ground_truth, g.space)
    print(result.summary())
    print("  before " + ate_before.describe())
    print("  after  " + ate_after.describe())
    if args.plot:
        from . import plotting

        print(f"wrote {plotting.plot_before_after(sim.graph.poses, g.poses, g.space, args.plot)}")
    return 0


def cmd_datasets(args) -> int:
    present = set(datasets.available())
    print(f"{'name':<20} {'space':<5} {'ground truth':<16} on disk")
    for name, spec in datasets.REGISTRY.items():
        print(
            f"{name:<20} {spec.space:<5} {str(spec.ground_truth or '-'):<16} "
            f"{'yes' if name in present else 'no'}"
        )
    print("\nFetch with: python3 tools/fetch_datasets.py --all")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="posegraph", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="parse a .g2o file and report its structure")
    p.add_argument("input")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("optimize", help="optimise a .g2o file")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None, help="write the optimised graph here")
    p.add_argument("--fix", type=int, default=None, help="pose id to hold fixed")
    p.add_argument("--plot", default=None, help="write a before/after PNG here")
    _add_solver_args(p)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("demo", help="run the synthetic front-end stub end to end")
    p.add_argument("--poses", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--se3", action="store_true")
    p.add_argument("--plot", default=None)
    _add_solver_args(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("datasets", help="list the benchmark datasets and what is on disk")
    p.set_defaults(func=cmd_datasets)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
