Benchmark datasets land here. Nothing in this directory is committed except this
file: the standard public pose graphs are mirrored under LGPL/GPL/BSD licences
that do not belong inside an MIT package, so they are fetched on demand and
verified by checksum instead.

    python3 tools/fetch_datasets.py --list
    python3 tools/fetch_datasets.py --all
    python3 tools/fetch_datasets.py --verify

Origins, licences, per-file SHA-256 values and the awkward details of each
format are in ../docs/datasets.md.
