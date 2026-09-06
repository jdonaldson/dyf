"""Does a frozen dyf partition survive distribution SHIFT, or only distribution GROWTH?

Same-parameter control: 4 conditions differing ONLY in how the base/stream split is
drawn. All hold base ~61% / stream ~39%, same tree params, same probe budget.

  random    i.i.d. split           (no shift -- the control)
  temporal  base = pre-2024        (growth -- what sec_sequence.py measured)
  ticker    disjoint companies     (shift)
  section   disjoint 10-Q sections (shift)

Queries are drawn from BOTH sides. Stream-side queries -- "does it serve content it
never saw" -- are the load-bearing comparison.

Replicated over N_SEEDS query draws, because a single 500-query draw has ~+-0.01 of
noise on the gap -- enough to invent an ordering among the three mild conditions that
does not survive replication.

Result (2026-08-01), stream-side gap @probe32, mean over 4 seeds:
  random ~0.00 | temporal ~+0.01 | ticker ~+0.02 | section ~+0.13
Only the section collapse exceeds the noise floor. random/temporal/ticker overlap.

TRAP: dirty% is ANTI-correlated with damage (~84% healthy, ~43% broken). Out-of-distribution
data piles into a corner of a partition that does not fit it. Never use delta size as a
health metric -- use the unseen-bucket rate.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

NQ = 500
PROBES = [32, 128]
K = 10
TARGET_BASE = 0.61
N_SEEDS = 4


def conditions(rng, E, T, SEC, Q, N):
    def by_group(labels):
        groups = list(dict.fromkeys(labels.tolist()))
        rng.shuffle(groups)
        counts = {g: int((labels == g).sum()) for g in groups}
        chosen, tot = set(), 0
        for g in groups:
            if tot / N >= TARGET_BASE:
                break
            chosen.add(g)
            tot += counts[g]
        return np.isin(labels, list(chosen))

    return {
        "random": rng.random(N) < TARGET_BASE,
        "temporal": Q <= "2023Q4",
        "ticker": by_group(T),
        "section": np.isin(SEC, ["risk_factors", "forward_looking"]),
    }


def run_once(seed, E, T, SEC, Q):
    rng = np.random.default_rng(seed)
    N = len(E)
    out = {}
    for name, mask in conditions(rng, E, T, SEC, Q, N).items():
        base_pool, stream_pool = np.where(mask)[0], np.where(~mask)[0]
        qb = rng.choice(base_pool, NQ, replace=False)
        qs = rng.choice(stream_pool, NQ, replace=False)
        held = np.zeros(N, bool)
        held[qb] = True
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

        row = {
            "dirty_frac": len(np.unique(a_all[stream_idx])) / NL,
            "unseen_rate": unseen / len(stream_idx),
        }
        for side, qidx in [("base", qb), ("stream", qs)]:
            QE = E[qidx]
            truth = S.exact_knn(QE, Ecur, k=K)
            for p in PROBES:
                rf = S.recall_at_k(S.ivf_search(QE, Ecur, a_fro, C_fro, p, K), truth)
                rn = S.recall_at_k(S.ivf_search(QE, Ecur, a_fresh, C_fresh, p, K), truth)
                row[f"{side}_p{p}_frozen"] = rf
                row[f"{side}_p{p}_gap"] = rn - rf
        out[name] = row
    return out


def main():
    E, D, T, SEC, Q = S.load()
    runs = []
    for s in range(N_SEEDS):
        t0 = time.time()
        runs.append(run_once(s, E, T, SEC, Q))
        print(
            f"seed {s} done [{time.time() - t0:.0f}s]  "
            + "  ".join(f"{k}={v['stream_p32_gap']:+.4f}" for k, v in runs[-1].items()),
            flush=True,
        )
        with open(os.path.join(S.CACHE, "shift_results.json"), "w") as f:
            json.dump(runs, f, indent=2)

    def agg(cond, key):
        v = np.array([r[cond][key] for r in runs])
        return v.mean(), v.std()

    print(f"\n=== stream-side queries, mean+-sd over {N_SEEDS} seeds ===")
    print(f"{'cond':<10}{'dirty%':>8}{'unseen%':>9}{'frozen@32':>12}{'gap@32':>18}{'gap@128':>18}")
    for c in ["random", "temporal", "ticker", "section"]:
        d, _ = agg(c, "dirty_frac")
        u, _ = agg(c, "unseen_rate")
        fr, _ = agg(c, "stream_p32_frozen")
        g32, s32 = agg(c, "stream_p32_gap")
        g128, s128 = agg(c, "stream_p128_gap")
        print(
            f"{c:<10}{100 * d:>7.1f}%{100 * u:>8.2f}%{fr:>12.4f}"
            f"{g32:>+11.4f} +-{s32:<5.4f}{g128:>+11.4f} +-{s128:<5.4f}"
        )

    noise = max(agg(c, "stream_p32_gap")[1] for c in ["random", "temporal", "ticker"])
    print(f"\nnoise floor (max sd among mild conditions) = {noise:.4f}")
    print("Only conditions whose gap exceeds ~2x this are distinguishable.")


if __name__ == "__main__":
    main()
