#!/usr/bin/env python3
"""Optional: optimise the same ``.g2o`` files with GTSAM and print both results.

This is a verification tool, not part of the package. GTSAM is **not** a
dependency of ``posegraph`` and nothing in ``src/`` imports it. The point of
this script is that a chi-squared number is only meaningful next to another
one, so the README's benchmark table is published alongside the numbers a
mature, independent implementation produces on the identical files.

GTSAM's ``NonlinearFactorGraph.error(values)`` is one half of
``sum e^T Omega e``, so everything below is twice it, which is the quantity
``posegraph`` reports.

GTSAM 4.2's prebuilt wheel is compiled against NumPy 1.x and crashes under
NumPy 2, so install it into its own environment::

    python3 -m venv /tmp/gtsam-check && . /tmp/gtsam-check/bin/activate
    pip install "numpy<2" gtsam
    python3 tools/gtsam_crosscheck.py

Both solvers are run with default settings. This is a consistency check, not a
competition, and neither side is tuned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posegraph import datasets  # noqa: E402
from posegraph.solver import SolverOptions, optimize  # noqa: E402


def _gtsam_run(path: Path, is3d: bool, max_iterations: int):
    import gtsam
    import numpy as np

    graph, initial = gtsam.readG2o(str(path), is3D=is3d)
    key = initial.keys()[0]
    if is3d:
        model = gtsam.noiseModel.Diagonal.Variances(np.full(6, 1e-6))
        graph.add(gtsam.PriorFactorPose3(key, initial.atPose3(key), model))
    else:
        model = gtsam.noiseModel.Diagonal.Variances(np.array([1e-6, 1e-6, 1e-8]))
        graph.add(gtsam.PriorFactorPose2(key, initial.atPose2(key), model))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    t0 = time.perf_counter()
    result = opt.optimize()
    return {
        "initial_chi2": 2.0 * graph.error(initial),
        "final_chi2": 2.0 * graph.error(result),
        "iterations": int(opt.iterations()),
        "seconds": time.perf_counter() - t0,
    }


def _ours_run(name: str, data_dir, max_iterations: int):
    g = datasets.load(name, data_dir)
    g.fix_pose(g.pose_ids[0])
    res = optimize(g, SolverOptions(method="lm", max_iterations=max_iterations))
    return {
        "initial_chi2": res.initial_chi2,
        "final_chi2": res.final_chi2,
        "iterations": res.iterations,
        "seconds": res.seconds,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="registry names (default: everything on disk)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args(argv)

    try:
        __import__("gtsam")
    except ImportError:
        print(
            "gtsam is not installed. This script is optional; see its docstring "
            "for the one-line install.",
            file=sys.stderr,
        )
        return 2

    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    names = args.names or datasets.available(data_dir)
    out = {}
    print(
        f"{'dataset':<18}{'ours final chi2':>18}{'gtsam final chi2':>18}"
        f"{'rel. difference':>17}"
    )
    for name in names:
        spec = datasets.REGISTRY.get(name)
        if spec is None:
            print(f"{name:<18}  unknown dataset", file=sys.stderr)
            continue
        path = datasets.dataset_path(name, data_dir)
        if not path.exists():
            print(f"{name:<18}  not downloaded, skipped")
            continue
        try:
            mine = _ours_run(name, data_dir, args.iterations)
            theirs = _gtsam_run(path, spec.space == "SE3", args.iterations)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print(f"{name:<18}  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        a, b = mine["final_chi2"], theirs["final_chi2"]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
        out[name] = {"posegraph": mine, "gtsam": theirs, "relative_difference": rel}
        print(f"{name:<18}{a:>18.6g}{b:>18.6g}{100 * rel:>16.3f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
