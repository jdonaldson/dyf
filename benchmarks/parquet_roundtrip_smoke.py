"""Real-scale smoke test for the Parquet round trip.

Reports export/import wall time, file sizes, and — the reason this exists — whether the
rebuilt index has the SAME leaf count as the source. Before build-param inheritance landed,
a 50k/384d product index went 842 leaves -> 14,385 and 48 MB -> 136 MB with nothing raised,
which is precisely the silent-reshape failure the round trip has to rule out.

Usage:
    dyf/.venv/bin/python benchmarks/parquet_roundtrip_smoke.py <index.dyf> [--keep]

Prints shapes and sizes only, never row contents — a production index may hold
customer-identifying fields.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dyf.lazy_index import LazyIndex  # noqa: E402
from dyf.parquet_io import from_parquet, to_parquet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--keep", action="store_true", help="leave the artifacts on disk")
    args = ap.parse_args()

    with LazyIndex(args.index) as idx:
        src_leaves, src_items, src_dim = idx.num_leaves, idx.total_items, idx.embedding_dim
        src_quant = idx.quantization
    src_bytes = os.path.getsize(args.index)
    print(
        f"source      {src_items:,} items  dim {src_dim}  {src_leaves:,} leaves  {src_quant}  {src_bytes / 1e6:.1f} MB"
    )

    tmp = tempfile.mkdtemp(prefix="dyf_pq_smoke_")
    parquet = os.path.join(tmp, "export.parquet")

    t = time.time()
    exp = to_parquet(args.index, parquet)
    print(
        f"to_parquet  {time.time() - t:.1f}s  {exp['rows']:,} rows  {len(exp['fields'])} fields  "
        f"{exp['bytes_out'] / 1e6:.1f} MB  ({exp['bytes_out'] / src_bytes:.2f}x source)"
    )

    back = os.path.join(tmp, "back.dyf")
    t = time.time()
    imp = from_parquet(parquet, back, quantization=src_quant)
    print(
        f"from_parquet {time.time() - t:.1f}s  {imp['rows']:,} rows  {len(imp['fields'])} fields  "
        f"{imp['bytes_out'] / 1e6:.1f} MB"
    )
    print(f"  build params {imp['build_params']}  inherited={imp['build_params_inherited']}")
    if imp["skipped"]:
        print(f"  skipped: {imp['skipped']}")

    ok = imp["leaves"] == src_leaves
    print(f"  leaves {src_leaves:,} -> {imp['leaves']:,}  {'OK (shape preserved)' if ok else 'MISMATCH'}")

    if args.keep:
        print(f"artifacts in {tmp}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
