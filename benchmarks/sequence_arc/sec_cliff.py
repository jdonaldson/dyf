"""Where is the coverage cliff?

sec_dose.py showed the frozen partition is unharmed by heavy sampling BIAS (93.5%
purity ~= no damage) but collapses at total ABSENCE (100% purity). This brackets the
untested gap: 0% -> 5% representation of the missing region, base size held fixed.

Result (2026-08-01): recovery is smooth over that range and essentially complete at
~5% representation (4,691 points of a 134k base -> gap +0.001, from +0.132 at zero).

  points from missing region:    0     47    188    938   1876   4691
  unseen%:                    9.34   6.56   8.04   5.33   3.46   1.97
  recall gap:                +.132  +.094  +.081  +.057  +.020  +.001

Requirement is COVERAGE, not proportion: every region needs a few percent presence at
fit time, but need not be proportionally represented.
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
NBASE = 134_000  # must be <= |IN| so f=0 is reachable
FRACS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def main():
    rng = np.random.default_rng(23)
    E, D, T, SEC, Q = S.load()
    N = len(E)
    IN_idx = np.where(np.isin(SEC, ["risk_factors", "forward_looking"]))[0]
    OUT_idx = np.where(~np.isin(SEC, ["risk_factors", "forward_looking"]))[0]
    print(f"IN={len(IN_idx)} OUT={len(OUT_idx)}  base held at {NBASE}\n")

    rows = []
    print(f"{'f_out':>8}{'n_from_OUT':>12}{'unseen%':>9}{'frozen@32':>11}{'fresh@32':>10}{'gap':>9}")
    print("-" * 59)
    for f in FRACS:
        n_out = int(round(f * len(OUT_idx)))
        take_out = rng.choice(OUT_idx, n_out, replace=False) if n_out else np.array([], dtype=np.int64)
        take_in = rng.choice(IN_idx, NBASE - n_out, replace=False)
        base_idx = np.concatenate([take_in, take_out]).astype(np.int64)
        in_base = np.zeros(N, bool)
        in_base[base_idx] = True

        out_rest = OUT_idx[~in_base[OUT_idx]]
        qs = rng.choice(out_rest, NQ, replace=False)
        held = np.zeros(N, bool)
        held[qs] = True
        stream_idx = np.where(~in_base & ~held)[0]
        cur = np.concatenate([base_idx, stream_idx])
        Ecur = E[cur]

        flat = S.build(E, base_idx)
        NL = S.n_leaves(flat)
        a_all, _ = S.route(E, flat)
        _, unseen = S.route(E[out_rest], flat)
        a_fro = a_all[cur]
        C_fro, _ = S.leaf_centroids(Ecur, a_fro, NL)

        flat_fresh = S.build(E, cur)
        a_fresh = S.fresh_assign(flat_fresh, len(cur))
        C_fresh, _ = S.leaf_centroids(Ecur, a_fresh, S.n_leaves(flat_fresh))

        QE = E[qs]
        truth = S.exact_knn(QE, Ecur, k=K)
        rf = S.recall_at_k(S.ivf_search(QE, Ecur, a_fro, C_fro, PROBE, K), truth)
        rn = S.recall_at_k(S.ivf_search(QE, Ecur, a_fresh, C_fresh, PROBE, K), truth)
        ur = unseen / len(out_rest)
        rows.append(dict(f=f, n_out=n_out, unseen=float(ur), frozen=rf, fresh=rn, gap=rn - rf))
        print(f"{f:>8.4f}{n_out:>12}{100 * ur:>8.2f}%{rf:>11.4f}{rn:>10.4f}{rn - rf:>+9.4f}", flush=True)
        with open(os.path.join(S.CACHE, "cliff_results.json"), "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
