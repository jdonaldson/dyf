"""Is the flat frozen-vs-fresh recall gap real, or is a fixed probe budget masking it?

Sweeps the probe budget at maximum drift (base = pre-2024, corpus = everything) and
reports recall against candidates ACTUALLY SCANNED, so the comparison is equal-work
rather than equal-probe.

Result (2026-08-01): the gap NARROWS to zero as probe rises (+0.008 at 32, -0.001 at
256) and scan cost is equal at equal probe. The flat gap was not an artifact.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

NQ = 500
K = 10
PROBES = [4, 8, 16, 32, 64, 128, 256, 512]


def main():
    rng = np.random.default_rng(0)
    E, D, T, SEC, Q = S.load()
    N = len(E)
    qsel = rng.choice(N, NQ, replace=False)
    held = np.zeros(N, bool)
    held[qsel] = True
    QE = E[qsel]
    cur = np.where(~held)[0]
    base_idx = cur[Q[cur] <= "2023Q4"]
    print(f"base={len(base_idx)} final={len(cur)} (+{100 * len(cur) / len(base_idx) - 100:.0f}% growth)")

    flat = S.build(E, base_idx)
    NL = S.n_leaves(flat)
    a_fro, unseen = S.route(E[cur], flat)
    C_fro, _ = S.leaf_centroids(E[cur], a_fro, NL)

    flat_fresh = S.build(E, cur)
    NLf = S.n_leaves(flat_fresh)
    a_fresh = S.fresh_assign(flat_fresh, len(cur))
    C_fresh, _ = S.leaf_centroids(E[cur], a_fresh, NLf)

    Ecur = E[cur]
    truth = S.exact_knn(QE, Ecur, k=K)
    print(f"frozen leaves={NL} fresh leaves={NLf} unseen-bucket={unseen}\n")

    print(f"{'probe':>6} | {'frozen R@10':>12} {'cands':>9} | {'fresh R@10':>11} {'cands':>9} | {'gap':>8}")
    print("-" * 68)
    rows = []
    for p in PROBES:
        rf = S.recall_at_k(S.ivf_search(QE, Ecur, a_fro, C_fro, p, K), truth)
        rn = S.recall_at_k(S.ivf_search(QE, Ecur, a_fresh, C_fresh, p, K), truth)
        cf = S.scan_cost(QE, a_fro, C_fro, p)
        cn = S.scan_cost(QE, a_fresh, C_fresh, p)
        rows.append((p, rf, cf, rn, cn))
        print(f"{p:>6} | {rf:>12.4f} {cf:>9.0f} | {rn:>11.4f} {cn:>9.0f} | {rn - rf:>+8.4f}", flush=True)

    print("\nequal-work (fresh recall interpolated at frozen's scan cost):")
    xs = np.array([r[4] for r in rows])
    ys = np.array([r[3] for r in rows])
    for p, rf, cf, _, _ in rows:
        at = np.interp(cf, xs, ys)
        print(f"  probe={p:>4}: frozen {rf:.4f} @ {cf:>6.0f} cands  vs fresh {at:.4f} @ same work  gap {at - rf:+.4f}")


if __name__ == "__main__":
    main()
