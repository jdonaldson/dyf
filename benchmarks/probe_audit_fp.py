"""Confirm the CONSTANT verdict on find_super_connectors is an audit false positive.

`classify` returns CONSTANT for a numeric array with a single distinct value. That
is the right rule for a SCORE array (no discrimination) and the wrong rule for an
INDEX array, where a one-element selection is a legitimate result.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "benchmarks")
import audit_public_api as A  # noqa: E402

from dyf import find_super_connectors  # noqa: E402


def main() -> None:
    X = A.load_real()
    print(f"audit fixture: {X.shape}")
    r = find_super_connectors(X)
    idx = np.asarray(r.indices)
    print(f"  indices        = {idx.tolist()}")
    print(f"  len={len(idx)}  n_unique={len(np.unique(idx))}")
    print(f"  classify(indices) = {A.classify(idx)}")
    print(f"  classify(result)  = {A.classify(r)}")
    q = np.asarray(r.quadrant)
    v, c = np.unique(q, return_counts=True)
    print(f"  quadrants = {dict(zip(v, c.tolist()))}")
    print(f"  global_centrality nonzero = {int((np.asarray(r.global_centrality) > 0).sum())}")
    print(f"  local_centrality  nonzero = {int((np.asarray(r.local_centrality) > 0).sum())}")
    print(f"\n  PAYLOAD_FIELDS = {sorted(A.PAYLOAD_FIELDS)}")
    if len(idx) == 1:
        print(
            "\n  -> FALSE POSITIVE: one super connector was found. An index array of "
            "length 1\n     has one distinct value by definition; CONSTANT is a rule "
            "for score arrays."
        )


if __name__ == "__main__":
    main()
