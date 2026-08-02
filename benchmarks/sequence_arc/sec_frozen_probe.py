"""Is fit_with_hyperplanes a TRUE frozen partition, or does it recenter on what it sees?

This is the load-bearing primitive for every sequence-of-dyfs claim. If bucket
assignment for the SAME points changed when other points were added, "dyf_t as the
foundation for dyf_{t+1}" would silently break.

Result (2026-08-01, dyf-rs 0.9.0): 100% on all four checks. Routing is a pure function
of (x, H), and reduces exactly to numpy sign bits, LSB-first.
"""

import os
import sys

import numpy as np
from dyf_rs import DensityClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

NB = 4


def main():
    E, *_ = S.load()
    A, B = E[:20000], E[20000:40000]
    dim = E.shape[1]

    c1 = DensityClassifier(embedding_dim=dim, num_bits=NB, seed=42)
    c1.fit_raw_pca(A)
    H = np.asarray(c1.get_hyperplanes(), dtype=np.float32)
    bA = np.asarray(c1.get_bucket_ids())
    print(f"hyperplanes {H.shape}")

    def buckets(X, arr):
        c = DensityClassifier(embedding_dim=dim, num_bits=NB, seed=42)
        c.fit_with_hyperplanes(X, H)
        return np.asarray(c.get_bucket_ids())[arr]

    bA2 = buckets(A, slice(None))
    print(f"[reproduce] fit_raw_pca vs fit_with_hyperplanes: {100 * (bA == bA2).mean():.2f}%")
    print(
        f"[frozen]    A's buckets with B present:          "
        f"{100 * (bA2 == buckets(np.concatenate([A, B]), slice(0, len(A)))).mean():.2f}%"
    )
    print(
        f"[order]     A's buckets placed after B:          "
        f"{100 * (bA2 == buckets(np.concatenate([B, A]), slice(len(B), None))).mean():.2f}%"
    )
    print(
        f"[subset]    first 100 routed alone:              "
        f"{100 * (buckets(A[:100], slice(None)) == bA2[:100]).mean():.2f}%"
    )

    proj = A @ H.T
    got = ((proj > 0).astype(np.int64) << np.arange(NB)).sum(1)
    print(f"[numpy]     sign>0 / LSB-first:                  {100 * (got == bA2).mean():.2f}%")


if __name__ == "__main__":
    main()
