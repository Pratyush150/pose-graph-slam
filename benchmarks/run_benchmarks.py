#!/usr/bin/env python3
"""Run the solver over every downloaded benchmark and write a results table.

Everything printed by this script is measured at run time. Nothing in the
README's benchmark table is typed by hand.

    python3 tools/fetch_datasets.py --all
    python3 benchmarks/run_benchmarks.py

Outputs land in ``benchmarks/output/``:

* ``results.json``  -- every measurement, machine-readable
* ``results.md``    -- the markdown table pasted into the README
* ``*_trajectory.png``, ``*_convergence.png``, ``*_sparsity.png``
* ``robust_comparison.png`` and ``robust_residuals.png``
* ``incremental.json`` -- windowed vs batch update timings
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posegraph import analysis, datasets, frontend_stub, linalg  # noqa: E402
from posegraph.incremental import compare_with_batch  # noqa: E402
from posegraph.solver import Problem, SolverOptions, optimize  # noqa: E402

DEFAULT_ORDER = [
    "tinyGrid3D",
    "smallGrid3D",
    "MIT",
    "CSAIL",
    "intel",
    "parking-garage",
    "manhattan",
    "sphere2500",
    "sphere_bignoise",
    "cubicle",
    "torus3D",
    "city10000",
    "grid3D",
    "ais2klinik",
    "rim",
]


def machine_info() -> Dict[str, str]:
    """Whatever we can actually read about the box, no guesses."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "processor": platform.processor() or "unknown",
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = "not installed"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info["sparse_backend_available"] = "scipy" if linalg.HAVE_SCIPY else "numpy fallback only"
    return info


@dataclass
class Row:
    """One benchmark row. Every field is measured."""

    dataset: str
    space: str
    poses: int
    edges: int
    loop_edges: int
    initial_chi2: float
    final_chi2: float
    iterations: int
    converged: bool
    seconds: float
    backend: str
    variable_blocks: int
    fill_natural: int
    fill_chosen: int
    fill_method: str
    fill_reduction_pct: float
    chi2_at_ground_truth: Optional[float] = None
    ate_before: Optional[float] = None
    ate_after: Optional[float] = None
    rpe_trans_after: Optional[float] = None
    rpe_rot_deg_after: Optional[float] = None
    note: str = ""


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 1e5 or abs(x) < 1e-3:
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    return str(x)


def markdown_table(rows: List[Row]) -> str:
    head = (
        "| dataset | space | poses | edges | loops | initial chi2 | final chi2 | "
        "chi2 at ground truth | iters | time (s) | ATE before (m) | ATE after (m) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r.dataset} | {r.space} | {r.poses} | {r.edges} | {r.loop_edges} | "
        f"{_fmt(r.initial_chi2)} | {_fmt(r.final_chi2)} | {_fmt(r.chi2_at_ground_truth)} | "
        f"{r.iterations} | {r.seconds:.2f} | {_fmt(r.ate_before)} | {_fmt(r.ate_after)} |\n"
        for r in rows
    )
    return head + body


def ordering_table(rows: List[Row]) -> str:
    head = (
        "| dataset | variable blocks | L blocks, natural order | L blocks, reordered | "
        "method | fill removed |\n|---|---:|---:|---:|---|---:|\n"
    )
    body = "".join(
        f"| {r.dataset} | {r.variable_blocks} | {r.fill_natural:,} | {r.fill_chosen:,} | "
        f"{r.fill_method} | {r.fill_reduction_pct:.1f}% |\n"
        for r in rows
    )
    return head + body


def run_one(name: str, args, out_dir: Path) -> Optional[Row]:
    spec = datasets.REGISTRY[name]
    print(f"\n=== {name} ({spec.space}) ", flush=True)
    graph = datasets.load(name, args.data_dir)
    if graph.num_poses > args.max_poses:
        print(f"    skipped: {graph.num_poses} poses exceeds --max-poses {args.max_poses}")
        return None
    graph.fix_pose(graph.pose_ids[0])
    print(f"    {graph.summary()}")

    before = graph.poses.copy()
    gt = None
    if spec.ground_truth:
        try:
            gt = datasets.load_ground_truth(spec.ground_truth, args.data_dir)
        except FileNotFoundError:
            print(f"    ground truth {spec.ground_truth} not downloaded, skipping ATE")

    ate_before = None
    chi2_gt = None
    if gt is not None and gt.shape[0] == graph.num_poses:
        ate_before = analysis.absolute_trajectory_error(before, gt, graph.space).rmse
        # cost of the *published ground-truth* trajectory under the same edges and
        # information matrices: a reference point that needs no external citation
        graph.poses = gt.copy()
        chi2_gt = graph.chi2()
        graph.poses = before.copy()
        print(f"    chi2 evaluated at the ground-truth trajectory: {chi2_gt:.6g}")

    opts = SolverOptions(
        method=args.method,
        max_iterations=args.iterations,
        kernel=args.kernel,
        ordering="auto",
        verbose=args.verbose,
    )
    result = optimize(graph, opts)
    print(f"    {result.summary()}  [{result.message}]")
    if result.ordering:
        print(f"    {result.ordering.describe()}")

    ate_after = rpe_t = rpe_r = None
    if gt is not None and gt.shape[0] == graph.num_poses:
        ate_after = analysis.absolute_trajectory_error(graph.poses, gt, graph.space).rmse
        rpe = analysis.relative_pose_error(graph.poses, gt, graph.space, delta=1)
        rpe_t, rpe_r = rpe.rmse, rpe.rotation_rmse_deg
        print(f"    ATE rmse {ate_before:.4f} -> {ate_after:.4f} m; RPE(1) {rpe.describe()}")

    o = result.ordering
    row = Row(
        dataset=name,
        space=graph.space,
        poses=graph.num_poses,
        edges=graph.num_edges,
        loop_edges=int(len(graph.loop_edges())),
        initial_chi2=result.initial_chi2,
        final_chi2=result.final_chi2,
        iterations=result.iterations,
        converged=result.converged,
        seconds=result.seconds,
        backend=result.backend,
        variable_blocks=o.n_blocks if o else 0,
        fill_natural=o.natural_nnz if o else 0,
        fill_chosen=o.chosen_nnz if o else 0,
        fill_method=o.chosen if o else "none",
        fill_reduction_pct=100.0 * o.reduction_vs_natural if o else 0.0,
        chi2_at_ground_truth=chi2_gt,
        ate_before=ate_before,
        ate_after=ate_after,
        rpe_trans_after=rpe_t,
        rpe_rot_deg_after=rpe_r,
        note="" if result.converged else f"stopped: {result.message}",
    )

    if not args.no_plots:
        from posegraph import plotting

        plotting.plot_before_after(
            before,
            graph.poses,
            graph.space,
            out_dir / f"{name}_trajectory.png",
            title=f"{name}: {graph.num_poses} poses, {graph.num_edges} edges",
            ground_truth=gt if (gt is not None and gt.shape[0] == graph.num_poses) else None,
        )
        plotting.plot_convergence(
            {args.method: result.chi2_history},
            out_dir / f"{name}_convergence.png",
            title=f"{name}: chi-squared per iteration",
            initial={args.method: result.initial_chi2},
        )
        if o is not None and graph.num_poses <= args.max_sparsity_poses:
            problem = Problem(graph)
            plotting.plot_sparsity(
                problem.block_adj,
                o.perm,
                out_dir / f"{name}_sparsity.png",
                title=f"{name}: block sparsity of H",
                fill=(o.natural_nnz, o.chosen_nnz),
            )
            plotting.plot_fill_pattern(
                problem.block_adj,
                o.perm,
                out_dir / f"{name}_fill.png",
                title=f"{name}: Cholesky factor of the information matrix",
            )
    return row


def run_robust_experiment(args, out_dir: Path) -> Dict[str, object]:
    """Inject false loop closures into a real benchmark and compare kernels."""
    name = args.robust_dataset
    print(f"\n=== robust-kernel experiment on {name} ", flush=True)
    spec = datasets.REGISTRY[name]
    base = datasets.load(name, args.data_dir)
    base.fix_pose(base.pose_ids[0])
    gt = None
    if spec.ground_truth:
        try:
            gt = datasets.load_ground_truth(spec.ground_truth, args.data_dir)
        except FileNotFoundError:
            gt = None

    corrupted, injected = frontend_stub.inject_false_loop_closures(
        base, args.false_closures, seed=args.seed, min_index_gap=50, translation_scale=20.0
    )
    print(f"    injected {len(injected)} false loop closures into {base.num_edges} edges")

    panels = []
    summary: Dict[str, object] = {
        "dataset": name,
        "false_loop_closures": int(len(injected)),
        "seed": args.seed,
        "kernels": {},
    }
    for label, kernel in (
        ("no kernel", None),
        ("huber", "huber"),
        ("dcs", "dcs"),
    ):
        g = corrupted.copy()
        res = optimize(
            g,
            SolverOptions(
                method="lm",
                max_iterations=args.iterations,
                kernel=kernel,
                kernel_delta=args.kernel_delta,
            ),
        )
        entry: Dict[str, object] = {
            "final_chi2": res.final_chi2,
            "iterations": res.iterations,
            "seconds": res.seconds,
        }
        if gt is not None and gt.shape[0] == g.num_poses:
            entry["ate_rmse"] = analysis.absolute_trajectory_error(g.poses, gt, g.space).rmse
        chi = g.edge_chi2()
        clean = np.setdiff1d(np.arange(g.num_edges), injected)
        entry["median_chi2_injected"] = float(np.median(chi[injected]))
        entry["median_chi2_genuine"] = float(np.median(chi[clean]))
        summary["kernels"][label] = entry
        panels.append((label, g.poses.copy()))
        print(f"    {label:<10} " + ", ".join(f"{k}={_fmt(v)}" for k, v in entry.items()))

    if not args.no_plots:
        from posegraph import plotting

        plotting.plot_kernel_comparison(
            panels,
            base.space,
            out_dir / "robust_comparison.png",
            title=(
                f"{name} with {len(injected)} false loop closures: "
                "the same graph, three cost functions"
            ),
            ground_truth=gt if (gt is not None and gt.shape[0] == base.num_poses) else None,
        )
        g = corrupted.copy()
        optimize(g, SolverOptions(method="lm", max_iterations=args.iterations, kernel="dcs"))
        plotting.plot_residual_histogram(
            g.edge_chi2(),
            out_dir / "robust_residuals.png",
            title=f"{name}: per-edge chi-squared after a DCS solve",
            highlight=injected,
        )
    return summary


def run_incremental_experiment(args, out_dir: Path) -> Dict[str, object]:
    """Windowed update versus full batch, on a real benchmark."""
    name = args.incremental_dataset
    print(f"\n=== incremental experiment on {name} ", flush=True)
    g = datasets.load(name, args.data_dir)
    g.fix_pose(g.pose_ids[0])
    optimize(g, SolverOptions(method="lm", max_iterations=args.iterations))

    loops = g.loop_edges()
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(loops, size=min(args.incremental_closures, loops.size), replace=False)
    closures = [
        (int(g.edge_i[k]), int(g.edge_j[k]), g.edge_z[k].copy(), g.edge_info[k].copy())
        for k in picked
    ]
    stripped = g.copy()
    keep = np.setdiff1d(np.arange(g.num_edges), picked)
    stripped.edge_i = g.edge_i[keep]
    stripped.edge_j = g.edge_j[keep]
    stripped.edge_z = g.edge_z[keep]
    stripped.edge_info = g.edge_info[keep]
    stripped.edge_tag = [g.edge_tag[k] for k in keep]
    stripped._edge_rows_cache = None

    out = compare_with_batch(
        stripped,
        closures,
        hops=args.hops,
        options=SolverOptions(method="lm", max_iterations=30),
    )
    out["dataset"] = name
    out["hops"] = args.hops
    print(
        f"    {len(closures)} loop closures replayed: "
        f"incremental {out['incremental_mean_ms']:.1f} ms/update, "
        f"batch {out['batch_mean_ms']:.1f} ms/update, "
        f"max pose difference {out['max_position_difference']:.4g} m"
    )
    return out


def _write_payload(out_dir, info, args, rows, robust, incremental, absent,
                   not_selected, failures) -> None:
    """Write ``results.json``. Called after every dataset, not only at the end."""
    payload = {
        "machine": info,
        "solver": {
            "method": args.method,
            "max_iterations": args.iterations,
            "kernel": args.kernel,
        },
        "generated_unix_time": time.time(),
        "results": [asdict(r) for r in rows],
        "robust_experiment": robust,
        "incremental_experiment": incremental,
        "not_downloaded": absent,
        "present_but_not_run": not_selected,
        "failures": failures,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="dataset names (default: all present)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "output"))
    ap.add_argument("--method", default="lm", choices=["lm", "gn", "dogleg"])
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--kernel-delta", type=float, default=None)
    ap.add_argument("--max-poses", type=int, default=20000)
    ap.add_argument("--max-sparsity-poses", type=int, default=6000)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--experiments-only",
        action="store_true",
        help="skip the dataset sweep and reuse the rows already in results.json, "
        "re-running only the robust-kernel and incremental experiments",
    )
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--no-incremental", action="store_true")
    ap.add_argument("--robust-dataset", default="manhattan")
    ap.add_argument("--incremental-dataset", default="intel")
    ap.add_argument("--incremental-closures", type=int, default=10)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--false-closures", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    args.data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    present = set(datasets.available(args.data_dir))
    if args.datasets:
        names = [n for n in args.datasets if n in datasets.REGISTRY]
        missing = [n for n in args.datasets if n not in datasets.REGISTRY]
        for n in missing:
            print(f"unknown dataset {n!r}", file=sys.stderr)
    else:
        names = [n for n in DEFAULT_ORDER if n in present]
        names += [n for n in present if n not in DEFAULT_ORDER]
    absent = [n for n in names if n not in present]
    names = [n for n in names if n in present]
    not_selected = [n for n in datasets.REGISTRY if n in present and n not in names]

    info = machine_info()
    print("machine:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if absent:
        print(f"\nnot downloaded, skipped: {', '.join(absent)}")
    if not_selected:
        print(f"on disk but not selected for this run: {', '.join(not_selected)}")
    if not names:
        print("\nNo datasets found. Run: python3 tools/fetch_datasets.py --all", file=sys.stderr)
        return 1

    rows: List[Row] = []
    failures: Dict[str, str] = {}
    if args.experiments_only:
        previous = out_dir / "results.json"
        if not previous.exists():
            print(f"{previous} does not exist; run the full sweep first", file=sys.stderr)
            return 1
        old_payload = json.loads(previous.read_text())
        rows = [Row(**r) for r in old_payload.get("results", [])]
        failures = dict(old_payload.get("failures", {}))
        absent = list(old_payload.get("not_downloaded", absent))
        not_selected = list(old_payload.get("present_but_not_run", not_selected))
        info = old_payload.get("machine", info)
        print(f"\nreusing {len(rows)} dataset rows from {previous}")
    else:
        for name in names:
            try:
                row = run_one(name, args, out_dir)
                if row is not None:
                    rows.append(row)
            except Exception as exc:  # noqa: BLE001 - report, do not hide
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {failures[name]}", file=sys.stderr)
                if args.verbose:
                    traceback.print_exc()
            # Checkpoint after every dataset. The large graphs can take minutes
            # and can exhaust memory on a small machine; losing the whole sweep
            # to the last one would be silly.
            _write_payload(out_dir, info, args, rows, None, None, absent,
                           not_selected, failures)

    robust = None
    if not args.no_robust and args.robust_dataset in present:
        try:
            robust = run_robust_experiment(args, out_dir)
        except Exception as exc:  # noqa: BLE001
            failures["robust_experiment"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {failures['robust_experiment']}", file=sys.stderr)

    incremental = None
    if not args.no_incremental and args.incremental_dataset in present:
        try:
            incremental = run_incremental_experiment(args, out_dir)
        except Exception as exc:  # noqa: BLE001
            failures["incremental_experiment"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {failures['incremental_experiment']}", file=sys.stderr)

    _write_payload(out_dir, info, args, rows, robust, incremental, absent,
                   not_selected, failures)

    md = ["# Benchmark results", ""]
    md.append("Measured by `benchmarks/run_benchmarks.py`. Timings are indicative of")
    md.append("scale: Python driving NumPy and SciPy on the machine below.")
    md.append("")
    md.append("```")
    for k, v in info.items():
        md.append(f"{k}: {v}")
    md.append("```")
    md.append("")
    md.append(markdown_table(rows))
    md.append("")
    md.append("## Fill-in reduction from variable reordering")
    md.append("")
    md.append(ordering_table(rows))
    if absent:
        md.append("")
        md.append("Not downloaded, therefore not run: " + ", ".join(absent) + ".")
    if not_selected:
        md.append("")
        md.append("On disk but not selected for this run: " + ", ".join(not_selected) + ".")
    if failures:
        md.append("")
        md.append("Failed:")
        for k, v in failures.items():
            md.append(f"* `{k}`: {v}")
    (out_dir / "results.md").write_text("\n".join(md) + "\n")

    if incremental is not None:
        (out_dir / "incremental.json").write_text(json.dumps(incremental, indent=2))

    print("\n" + markdown_table(rows))
    print(ordering_table(rows))
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
