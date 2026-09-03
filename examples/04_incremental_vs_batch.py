#!/usr/bin/env python3
"""Cost of a windowed update against a full batch re-solve.

Both variants start from a map that is already consistent -- which is the
situation an incremental optimiser is actually for -- then take a few of its own
loop closures out and add them back one at a time.

    python3 examples/04_incremental_vs_batch.py
    python3 examples/04_incremental_vs_batch.py --dataset intel --hops 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posegraph import datasets, frontend_stub  # noqa: E402
from posegraph.incremental import compare_with_batch  # noqa: E402
from posegraph.solver import SolverOptions, optimize  # noqa: E402


def strip_edges(graph, indices):
    """Remove the given edges, returning a copy plus the removed constraints."""
    removed = [
        (int(graph.edge_i[k]), int(graph.edge_j[k]),
         graph.edge_z[k].copy(), graph.edge_info[k].copy())
        for k in indices
    ]
    keep = np.setdiff1d(np.arange(graph.num_edges), indices)
    g = graph.copy()
    g.edge_i, g.edge_j = graph.edge_i[keep], graph.edge_j[keep]
    g.edge_z, g.edge_info = graph.edge_z[keep], graph.edge_info[keep]
    g.edge_tag = [graph.edge_tag[k] for k in keep]
    g._edge_rows_cache = None
    return g, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None, help="registry name; default is synthetic")
    ap.add_argument("--poses", type=int, default=600)
    ap.add_argument("--closures", type=int, default=10)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dataset:
        g = datasets.load(args.dataset)
        g.fix_pose(g.pose_ids[0])
        label = args.dataset
    else:
        g = frontend_stub.simulate_se2(
            num_poses=args.poses, turn_every=25, min_index_gap=40, seed=args.seed
        ).graph
        label = "synthetic circuit"

    # Start from a converged map: that is the state an incremental optimiser
    # maintains, and comparing against a batch solve from a bad initial guess
    # would flatter it for the wrong reason.
    optimize(g, SolverOptions(method="lm", max_iterations=200))

    loops = g.loop_edges()
    if loops.size == 0:
        print("this graph has no loop closures to replay", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(loops, size=min(args.closures, loops.size), replace=False)
    base, closures = strip_edges(g, picked)

    print(f"{label}: {base.num_poses} poses, {base.num_edges} edges after removal")
    print(f"replaying {len(closures)} loop closures, window radius {args.hops} hops\n")

    out = compare_with_batch(
        base, closures, hops=args.hops, options=SolverOptions(method="lm", max_iterations=30)
    )
    print(f"{'update':>7}{'window poses':>15}{'incremental ms':>17}{'batch ms':>12}")
    for k, (w, ti, tb) in enumerate(
        zip(out["windows"], out["incremental_times"], out["batch_times"]), start=1
    ):
        print(f"{k:>7}{w:>15}{1e3 * ti:>17.1f}{1e3 * tb:>12.1f}")
    speedup = out["batch_mean_ms"] / max(out["incremental_mean_ms"], 1e-9)
    print(
        f"\nmean per update: incremental {out['incremental_mean_ms']:.1f} ms, "
        f"batch {out['batch_mean_ms']:.1f} ms ({speedup:.1f}x)"
    )
    print(
        f"final chi2: incremental {out['incremental_chi2']:.6g}, "
        f"batch {out['batch_chi2']:.6g}"
    )
    print(
        "largest per-pose position difference between the two: "
        f"{out['max_position_difference']:.6g} m "
        f"(RMS {out['rms_position_difference']:.6g} m)"
    )
    print(
        "\nA small window is cheap but only correct while the correction stays inside\n"
        "it. Widen --hops until the position difference stops shrinking, or run a\n"
        "batch pass periodically."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
