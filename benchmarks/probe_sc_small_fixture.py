"""Why does find_super_connectors read CONSTANT on the 400-point audit fixture?

The public-API audit uses 400 real SEC sections. `_derive_num_bits(400, 20)` gives
3 bits = 8 buckets, ~50 points each, which should clear the dense gate. Find out
what is actually constant, and whether the derived resolution is right this small.
"""

from __future__ import annotations

import os

import numpy as np

from dyf import find_super_connectors
from dyf.rag import _derive_num_bits


def load_sec(n: int) -> np.ndarray | None:
    """Same loader the public-API audit uses (benchmarks/sequence_arc/sec_seqlib.py)."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_arc"))
    try:
        import sec_seqlib as S
    except Exception:
        return None
    E, *_ = S.load()
    rng = np.random.default_rng(42)
    pick = rng.choice(len(E), min(n, len(E)), replace=False)
    emb = np.asarray(E[np.sort(pick)], dtype=np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def describe(emb: np.ndarray, label: str) -> None:
    n = len(emb)
    bits = _derive_num_bits(n, 20)
    res = find_super_connectors(emb)
    quad = np.asarray(res.quadrant)
    gc = np.asarray(res.global_centrality)
    lc = np.asarray(res.local_centrality)
    vals, cnts = np.unique(quad, return_counts=True)
    print(f"\n{label}: n={n} derived_bits={bits} ({2**bits} buckets, ~{n / 2**bits:.0f}/bucket)")
    print(f"  quadrants: {dict(zip(vals, cnts.tolist()))}")
    print(f"  global_centrality: uniq={len(np.unique(gc))} min={gc.min()} max={gc.max()}")
    print(f"  local_centrality:  uniq={len(np.unique(lc))} min={lc.min()} max={lc.max()}")
    print(f"  super={len(res.indices)}  thresholds g={res.global_threshold:.2f} l={res.local_threshold:.2f}")
    if len(vals) == 1:
        print("  -> CONSTANT: every point got the same quadrant label")


def main() -> None:
    sec = load_sec(4000)
    if sec is None:
        print("SEC corpus unavailable; set SEC_DYF")
        return
    for n in (400, 800, 2000, 4000):
        describe(sec[:n], "SEC")

    print("\nSame sizes with a floor on bucket occupancy (bits chosen so ~>=100/bucket)")
    for n in (400, 800, 2000, 4000):
        emb = sec[:n]
        bits = max(2, int(np.log2(max(len(emb) / 100, 4))))
        res = find_super_connectors(emb, global_num_bits=bits, facet_num_bits=max(2, bits - 2))
        quad = np.asarray(res.quadrant)
        vals, cnts = np.unique(quad, return_counts=True)
        print(f"  n={len(emb):>5} bits={bits} super={len(res.indices):>4} quadrants={dict(zip(vals, cnts.tolist()))}")


if __name__ == "__main__":
    main()
