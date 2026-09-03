#!/usr/bin/env python3
"""Download the benchmark pose graphs into ``data/`` and verify their checksums.

The datasets are not redistributed with this repository: the mirrors they come
from are LGPL/GPL/BSD licensed and that is not compatible with shipping them
inside an MIT-licensed package. This script fetches them on demand and checks
the SHA-256 of every file against the value recorded in
``posegraph.datasets``, so a benchmark run is reproducible even though the
bytes are not in git.

Usage::

    python3 tools/fetch_datasets.py --list
    python3 tools/fetch_datasets.py intel sphere2500
    python3 tools/fetch_datasets.py --all
    python3 tools/fetch_datasets.py --verify
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posegraph import datasets  # noqa: E402


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1.0:.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _specs():
    out = dict(datasets.REGISTRY)
    out.update(datasets.GROUND_TRUTH)
    return out


def list_datasets(data_dir: Path) -> None:
    print(f"{'name':<20} {'space':<5} {'size':>9}  {'on disk':<8} source")
    for name, spec in _specs().items():
        path = data_dir / spec.filename
        state = "yes" if path.exists() else "no"
        space = getattr(spec, "space", "")
        print(f"{name:<20} {space:<5} {_size(spec.size_bytes):>9}  {state:<8} {spec.url}")
    print()
    print("Ground truth is available for: " + ", ".join(
        f"{n} -> {s.ground_truth}" for n, s in datasets.REGISTRY.items() if s.ground_truth
    ))


def fetch_one(name: str, data_dir: Path, force: bool = False, timeout: int = 300) -> bool:
    specs = _specs()
    if name not in specs:
        print(f"unknown dataset {name!r}", file=sys.stderr)
        return False
    spec = specs[name]
    path = data_dir / spec.filename
    if path.exists() and not force:
        digest = datasets.sha256_of(path)
        if digest == spec.sha256:
            print(f"ok      {name:<20} already present and verified")
            return True
        print(f"stale   {name:<20} checksum mismatch, re-downloading")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetch   {name:<20} {_size(spec.size_bytes):>9}  {spec.url}")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(spec.url, timeout=timeout) as resp, open(tmp, "wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"FAILED  {name:<20} {exc}", file=sys.stderr)
        return False
    digest = datasets.sha256_of(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        print(
            f"FAILED  {name:<20} sha256 mismatch\n"
            f"        expected {spec.sha256}\n        got      {digest}",
            file=sys.stderr,
        )
        return False
    tmp.replace(path)
    print(f"ok      {name:<20} {_size(path.stat().st_size):>9}  sha256 verified")
    return True


def verify(data_dir: Path) -> int:
    bad = 0
    for name, spec in _specs().items():
        path = data_dir / spec.filename
        if not path.exists():
            continue
        digest = datasets.sha256_of(path)
        status = "ok" if digest == spec.sha256 else "MISMATCH"
        if digest != spec.sha256:
            bad += 1
        print(f"{status:<9} {name:<20} {digest}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="dataset names to fetch")
    ap.add_argument("--all", action="store_true", help="fetch every registered dataset")
    ap.add_argument("--se2", action="store_true", help="fetch the SE(2) datasets")
    ap.add_argument("--se3", action="store_true", help="fetch the SE(3) datasets")
    ap.add_argument("--truth", action="store_true", help="fetch the ground-truth files")
    ap.add_argument("--list", action="store_true", help="list what is registered")
    ap.add_argument("--verify", action="store_true", help="checksum what is on disk")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--data-dir", default=None, help="target directory (default: <repo>/data)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else datasets.default_data_dir()

    if args.list:
        list_datasets(data_dir)
        return 0
    if args.verify:
        return 1 if verify(data_dir) else 0

    names = list(args.names)
    if args.all:
        names = list(datasets.REGISTRY) + list(datasets.GROUND_TRUTH)
    else:
        if args.se2:
            names += [n for n, s in datasets.REGISTRY.items() if s.space == "SE2"]
        if args.se3:
            names += [n for n, s in datasets.REGISTRY.items() if s.space == "SE3"]
        if args.truth:
            names += list(datasets.GROUND_TRUTH)
    if not names:
        ap.print_help()
        return 2

    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    failures = [n for n in seen if not fetch_one(n, data_dir, force=args.force)]
    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{len(seen)} dataset(s) ready in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
