"""Dose-response: does the unseen-bucket rate actually TRACK frozen-partition damage?

sec_shift.py suggested unseen% predicts recall loss, but n=4 cannot support a threshold.
Here shift is a continuous dial: base is sampled with sections {risk_factors,
forward_looking} oversampled by weight w. w=1 is no shift; w=inf is the full section
split. Base size held at ~61% so only PURITY varies.

Result (2026-08-01): FALSIFIES the linear story. Pearson r=+0.97 but RANK correlation
r=+0.18 -- the entire correlation is one point. Purity 59% -> 93.5% produces no trend
(gap wobbles around +0.005); only total absence (100%) collapses it to +0.122.

Sampling BIAS is harmless; missing COVERAGE is not. This sweep looks like a cliff only
because it has no samples between 93.5% and 100% purity -- sec_cliff.py brackets that
gap and resolves it into a steep but continuous knee.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

NQ = 500
K = 10
PROBE = 32
TARGET = 0.61
WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]


def main():
    rng = np.random.default_rng(11)
    E, D, T, SEC, Q = S.load()
    N = len(E)
    IN = np.isin(SEC, ["risk_factors", "forward_looking"])

    def base_mask(w):
        if np.isinf(w):
            return IN.copy()
        p = np.where(IN, w, 1.0).astype(np.float64)
        lo, hi = 0.0, 1.0 / p.min()
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.clip(p * mid, 0, 1).mean() < TARGET:
                lo = mid
            else:
                hi = mid
        return rng.random(N) < np.clip(p * ((lo + hi) / 2), 0, 1)

    rows = []
    print(f"{'w':>6}{'purity':>9}{'n_base':>9}{'dirty%':>8}{'unseen%':>9}{'frozen@32':>11}{'fresh@32':>10}{'gap':>9}")
    print("-" * 71)
    for w in WEIGHTS:
        m = base_mask(w)
        base_pool, stream_pool = np.where(m)[0], np.where(~m)[0]
        qs = rng.choice(stream_pool, NQ, replace=False)
        held = np.zeros(N, bool)
        held[qs] = True
        base_idx = base_pool[~held[base_pool]]
        stream_idx = stream_pool[~held[stream_pool]]
        cur = np.concatenate([base_idx, stream_idx])
        Ecur = E[cur]

        flat = S.build(E, base_idx)
        NL = S.n_leaves(flat)
        a_all, _ = S.route(E, flat)
        _, unseen = S.route(E[stream_idx], flat)
        a_fro = a_all[cur]
        C_fro, _ = S.leaf_centroids(Ecur, a_fro, NL)

        flat_fresh = S.build(E, cur)
        a_fresh = S.fresh_assign(flat_fresh, len(cur))
        C_fresh, _ = S.leaf_centroids(Ecur, a_fresh, S.n_leaves(flat_fresh))

        QE = E[qs]
        truth = S.exact_knn(QE, Ecur, k=K)
        rf = S.recall_at_k(S.ivf_search(QE, Ecur, a_fro, C_fro, PROBE, K), truth)
        rn = S.recall_at_k(S.ivf_search(QE, Ecur, a_fresh, C_fresh, PROBE, K), truth)
        purity = float(IN[base_idx].mean())
        dirty = len(np.unique(a_all[stream_idx])) / NL
        ur = unseen / len(stream_idx)
        rows.append(
            dict(
                w=float(w),
                purity=purity,
                n_base=len(base_idx),
                dirty=dirty,
                unseen=float(ur),
                frozen=rf,
                fresh=rn,
                gap=rn - rf,
            )
        )
        print(
            f"{w:>6.1f}{100 * purity:>8.1f}%{len(base_idx):>9}{100 * dirty:>7.1f}%"
            f"{100 * ur:>8.2f}%{rf:>11.4f}{rn:>10.4f}{rn - rf:>+9.4f}",
            flush=True,
        )
        with open(os.path.join(S.CACHE, "dose_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    u = np.array([r["unseen"] for r in rows])
    g = np.array([r["gap"] for r in rows])
    dy = np.array([r["dirty"] for r in rows])
    rank = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    print(f"\npearson(unseen%, gap) = {np.corrcoef(u, g)[0, 1]:+.4f}  (n={len(rows)})")
    print(f"pearson(dirty%,  gap) = {np.corrcoef(dy, g)[0, 1]:+.4f}")
    print(
        f"RANK corr(unseen%, gap) = {np.corrcoef(rank(u), rank(g))[0, 1]:+.4f}"
        "   <-- if this is near zero, the pearson is one-point leverage"
    )


if __name__ == "__main__":
    main()
