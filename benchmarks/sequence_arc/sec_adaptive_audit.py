"""Does dyf's SHIPPED adaptive probing actually help? Audit of the real code path.

`sec_adaptive_probe.py` tested margin as a rank-allocation signal and found ~0 effect, but
that is NOT `LazyIndex`'s mechanism. The shipped logic (`lazy_index.py:_resolve_nprobe`)
interpolates a probe count from the primary path's minimum routing margin:

    margin <= margin_lo (0.01)  -> max_probes (5)
    margin >= margin_hi (0.10)  -> min_probes (1)
    between                     -> linear

Two things make this worth auditing rather than trusting:

1. The thresholds are ABSOLUTE margins, but |projection| scales with the embedding norm and
   the hyperplane normalisation. Measured on this corpus, per-PC projections have sd ~0.19,
   so a 0.1 cutoff may sit low in the margin distribution -- in which case "auto" resolves
   to min_probes for nearly every query and the feature is a no-op dressed as adaptation.
2. The default range is 1..5 probes. Recall curves here need hundreds of probes to reach
   0.9, so even perfect allocation inside 1..5 can only move the very cheapest end.

WHAT IS MEASURED, through the real `LazyIndex.search` with `return_routing=True`:

  * the distribution of `min_margin` and the nprobe it resolves to, so "does auto ever
    choose anything but the minimum" is answered directly;
  * recall@10 and candidates_scored for "auto" versus a sweep of FIXED nprobe, on the same
    index and queries. The question is whether auto lands ABOVE the fixed-nprobe frontier
    (genuinely allocating), ON it (just picking a point, no value), or BELOW it (harmful).

Both backends are exercised, since the rust kernel reimplements the traversal and could
resolve nprobe differently from the python reference path.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

from dyf.dyf_tree import build_dyf_tree  # noqa: E402
from dyf.lazy_index import AdaptiveProbeConfig, LazyIndex, write_lazy_index  # noqa: E402

N_CORPUS = 100000
NQ = 400
K = 10
FIXED = (1, 2, 3, 5, 8, 16, 32, 64, 128)
SEED = 42


def recall(got, truth):
    return len(set(int(x) for x in got) & set(int(x) for x in truth)) / K


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    if len(E) > N_CORPUS:
        E = np.ascontiguousarray(E[rng.choice(len(E), N_CORPUS, replace=False)])
    qi = rng.choice(len(E), NQ, replace=False)
    Qe = E[qi]
    print(f"corpus {E.shape}, {NQ} queries", flush=True)
    truth = S.exact_knn(Qe, E, k=K)

    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "adaptive_audit.dyf")
    tree = build_dyf_tree(E, max_depth=4, num_bits=4, min_leaf_size=16, seed=SEED)
    write_lazy_index(tree, E, path)
    idx = LazyIndex(path)
    print(f"index: {os.path.getsize(path) / 1e6:.0f} MB", flush=True)

    out = {}
    for backend in ("python", "rust"):
        print(f"\n=== backend={backend} ===", flush=True)
        rows = {}

        # --- what does "auto" actually decide? -------------------------------------
        margins, nprobes, recs, cands = [], [], [], []
        t0 = time.time()
        for i in range(NQ):
            try:
                r = idx.search(Qe[i], k=K, nprobe="auto", return_routing=True, backend=backend)
            except Exception as e:  # noqa: BLE001
                print(f"  auto unsupported on {backend}: {type(e).__name__}: {e}")
                margins = None
                break
            ro = r.routing or {}
            margins.append(ro.get("min_margin"))
            nprobes.append(ro.get("adaptive_nprobe", len(ro.get("leaves_probed", []))))
            cands.append(ro.get("candidates_scored", 0))
            recs.append(recall(r.indices, truth[i]))
        if margins is None:
            continue
        auto_s = time.time() - t0
        mv = np.array([m for m in margins if m is not None], dtype=float)
        nv = np.array([n for n in nprobes if n is not None], dtype=float)
        cfg = AdaptiveProbeConfig()
        print(
            f"  min_margin: median {np.median(mv):.4f}, p10 {np.percentile(mv, 10):.4f}, "
            f"p90 {np.percentile(mv, 90):.4f}  (thresholds lo={cfg.margin_lo} hi={cfg.margin_hi})"
        )
        print(
            f"  fraction of queries at/above margin_hi (-> min_probes): "
            f"{100 * (mv >= cfg.margin_hi).mean():.1f}%; at/below margin_lo (-> max_probes): "
            f"{100 * (mv <= cfg.margin_lo).mean():.1f}%"
        )
        vals, counts = np.unique(nv, return_counts=True)
        print(f"  resolved nprobe distribution: {dict(zip(vals.astype(int).tolist(), counts.tolist()))}")
        print(
            f"  auto: recall={np.mean(recs):.4f} candidates={np.mean(cands):.0f} [{auto_s:.0f}s]",
            flush=True,
        )
        rows["auto"] = {"recall": float(np.mean(recs)), "candidates": float(np.mean(cands))}

        # --- the fixed-nprobe frontier it must beat --------------------------------
        print(f"  {'nprobe':>8}{'recall':>10}{'candidates':>13}")
        for p in FIXED:
            rr, cc = [], []
            for i in range(NQ):
                r = idx.search(Qe[i], k=K, nprobe=p, return_routing=True, backend=backend)
                rr.append(recall(r.indices, truth[i]))
                cc.append((r.routing or {}).get("candidates_scored", 0))
            rows[f"fixed{p}"] = {"recall": float(np.mean(rr)), "candidates": float(np.mean(cc))}
            print(f"  {p:>8}{np.mean(rr):>10.4f}{np.mean(cc):>13.0f}", flush=True)

        # --- is auto on, above, or below the frontier? -----------------------------
        fx = sorted(
            ((rows[f"fixed{p}"]["candidates"], rows[f"fixed{p}"]["recall"]) for p in FIXED),
            key=lambda t: t[0],
        )
        ac, ar = rows["auto"]["candidates"], rows["auto"]["recall"]
        interp = float(np.interp(ac, [c for c, _ in fx], [v for _, v in fx]))
        delta = ar - interp
        verdict = "ABOVE (allocating)" if delta > 0.005 else ("BELOW (harmful)" if delta < -0.005 else "ON (no value)")
        print(
            f"  auto at {ac:.0f} candidates: recall {ar:.4f} vs fixed-frontier {interp:.4f} -> {delta:+.4f}  {verdict}"
        )
        rows["_frontier_delta"] = delta
        out[backend] = rows

    os.unlink(path)
    p = os.path.join(S.CACHE, "adaptive_audit_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
