"""Is `find_super_connectors` structurally unable to return anything at small n?

A "Super Connector" requires `high_global & high_local`. Local centrality is only
computed inside DENSE buckets: `counts > max(percentile(counts, dense_percentile),
min_bucket_size)`. With `global_num_bits=12` that is 4096 buckets, so with
`min_bucket_size=20` the corpus needs roughly 20*4096 = 82k points before any
bucket qualifies -- unless the data is clustered enough to pile up.

If that is right, the defaults have a *scale* dependency the same way
`bridge_threshold` had an *anisotropy* dependency (KNOWN_ISSUES #5): the function
silently returns nothing outside a regime that is nowhere documented.

Sweeps n and num_bits, reporting how many buckets clear the dense gate and
whether any super connector is produced.
"""

from __future__ import annotations

import numpy as np

from dyf import DensityClassifier, find_super_connectors


def isotropic(n: int, dim: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def clustered(n: int, dim: int = 64, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per = n // n_clusters
    out = []
    for i in range(n_clusters):
        c = np.zeros(dim, dtype=np.float32)
        c[(i * 10) % dim : (i * 10) % dim + 10] = 1.0
        c /= np.linalg.norm(c)
        pts = c + rng.standard_normal((per, dim)).astype(np.float32) * 0.1
        out.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    e = np.vstack(out)
    return e[:n] if len(e) >= n else e


def dense_gate(emb: np.ndarray, num_bits: int, min_bucket_size: int, dense_pct: float = 75) -> tuple[int, int, float]:
    """Return (n_dense_buckets, n_occupied_buckets, max_bucket_size)."""
    clf = DensityClassifier(embedding_dim=emb.shape[1], num_bits=num_bits, seed=42)
    clf.fit(emb)
    counts = np.bincount(clf.get_bucket_ids())
    occupied = counts[counts > 0]
    thresh = np.percentile(occupied, dense_pct)
    n_dense = int((counts > max(thresh, min_bucket_size)).sum())
    return n_dense, len(occupied), int(occupied.max())


def main() -> None:
    print("Dense-bucket gate at the DEFAULT global_num_bits=12 (4096 buckets)")
    print("A super connector is impossible when n_dense == 0.\n")
    print(f"{'data':<10} {'n':>7} {'occupied':>9} {'max_bkt':>8} {'n_dense':>8} {'super':>6} {'quadrants'}")
    for kind, gen in (("isotropic", isotropic), ("clustered", clustered)):
        for n in (500, 2000, 8000, 30000):
            emb = gen(n)
            n_dense, n_occ, mx = dense_gate(emb, 12, 20)
            # explicit 12/10 = the pre-fix shipped default, kept so this table stays a record
            # of the broken behaviour after the default changed to None
            res = find_super_connectors(emb, global_num_bits=12, facet_num_bits=10)
            quad = np.asarray(res.quadrant)
            vals, cnts = np.unique(quad, return_counts=True)
            short = {
                "Regular": "reg",
                "Minor Bridge": "min",
                "Cross-Domain": "cross",
                "Domain Specialist": "spec",
                "Super Connector": "SUPER",
            }
            qs = " ".join(f"{short.get(v, v)}={c}" for v, c in zip(vals, cnts.tolist()))
            print(f"{kind:<10} {n:>7} {n_occ:>9} {mx:>8} {n_dense:>8} {len(res.indices):>6} {qs}")

    print("\nSHIPPED DEFAULT after the fix (global_num_bits=None -> _derive_num_bits)")
    print(f"{'data':<10} {'n':>7} {'bits':>5} {'occupied':>9} {'max_bkt':>8} {'n_dense':>8} {'super':>6}")
    from dyf.rag import _derive_num_bits

    for kind, gen in (("isotropic", isotropic), ("clustered", clustered)):
        for n in (500, 2000, 8000, 30000):
            emb = gen(n)
            bits = _derive_num_bits(n, 20)
            n_dense, n_occ, mx = dense_gate(emb, bits, 20)
            res = find_super_connectors(emb)  # defaults only
            print(f"{kind:<10} {n:>7} {bits:>5} {n_occ:>9} {mx:>8} {n_dense:>8} {len(res.indices):>6}")

    print("\nBridgeIndex end-to-end on the test fixture (500 pts) — was 0 super connectors")
    from dyf import BridgeIndex

    idx = BridgeIndex(n_anchors=50)
    idx.fit(isotropic(500), verbose=False)
    sc = idx.get_super_connectors()
    print(
        f"  resolved num_bits={idx.global_num_bits}/{idx.facet_num_bits}  "
        f"super={len(sc.indices)}  anchors={len(idx.get_anchors())}"
    )


if __name__ == "__main__":
    main()
