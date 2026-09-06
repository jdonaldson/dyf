"""Append-only sequence of dyfs: dirty-leaf fraction + frozen-vs-fresh search, per quarter.

(a) how many leaf batches change per step -> delta-encoding ceiling
(b) recall of a frozen partition vs a full rebuild -> how often you must refit

Null for (a) is a PERMUTATION null: same number of points, sampled uniformly from the
corpus (real points, real spatial distribution), time association destroyed. Plus the
analytic i.i.d. null sum_l (1 - size_l/N)^n_new.

Result (2026-08-01): a big quarter dirties ~44% of leaves vs ~5400 expected under the
null -- only ~10% better than random arrival. Recall gap frozen-vs-fresh stays ~+0.008
across 9 quarters and +64% growth.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

BASE_Q = "2023Q4"
NQ = 500
PROBE = 32
K = 10
NULL_REPS = 20


def main():
    rng = np.random.default_rng(0)
    E, D, T, SEC, Q = S.load()
    N = len(E)

    qsel = rng.choice(N, NQ, replace=False)
    held = np.zeros(N, bool)
    held[qsel] = True
    QE = E[qsel]
    pool = np.where(~held)[0]

    steps = [q for q in sorted(set(Q.tolist())) if q > BASE_Q]
    base_idx = pool[Q[pool] <= BASE_Q]
    print(f"base({BASE_Q}) n={len(base_idx)} steps={len(steps)} queries={NQ}", flush=True)

    t0 = time.time()
    flat = S.build(E, base_idx)
    NL = S.n_leaves(flat)
    print(f"base tree: {len(flat)} nodes, {NL} leaves, {time.time() - t0:.1f}s", flush=True)

    parent = np.full(len(flat), -1, np.int64)
    for i, n in enumerate(flat):
        for c in n["children"]:
            parent[c] = i
    leaf_node = np.full(NL, -1, np.int64)
    for i, n in enumerate(flat):
        if n["leaf_id"] >= 0:
            leaf_node[n["leaf_id"]] = i

    assign_all, unseen_all = S.route(E, flat)
    base_sizes = np.bincount(assign_all[base_idx], minlength=NL).astype(np.float64)

    def dirty_nodes(leaves):
        seen = set()
        for lf in leaves:
            n = leaf_node[lf]
            while n != -1 and n not in seen:
                seen.add(int(n))
                n = parent[n]
        return len(seen)

    results, cum = [], list(base_idx)
    for si, qtr in enumerate(steps):
        new_idx = pool[Q[pool] == qtr]
        new_leaves = np.unique(assign_all[new_idx])
        obs = len(new_leaves)

        null_d = np.array(
            [len(np.unique(assign_all[rng.choice(N, len(new_idx), replace=False)])) for _ in range(NULL_REPS)]
        )
        p = base_sizes / base_sizes.sum()
        iid = NL - np.sum((1.0 - p) ** len(new_idx))

        cum.extend(new_idx.tolist())
        cur = np.array(cum)
        Ecur = E[cur]

        a_fro = assign_all[cur]
        C_fro, cnt = S.leaf_centroids(Ecur, a_fro, NL)
        t1 = time.time()
        flat_fresh = S.build(E, cur)
        build_s = time.time() - t1
        NLf = S.n_leaves(flat_fresh)
        a_fresh = S.fresh_assign(flat_fresh, len(cur))
        C_fresh, _ = S.leaf_centroids(Ecur, a_fresh, NLf)

        truth = S.exact_knn(QE, Ecur, k=K)
        r_fro = S.recall_at_k(S.ivf_search(QE, Ecur, a_fro, C_fro, PROBE, K), truth)
        r_fresh = S.recall_at_k(S.ivf_search(QE, Ecur, a_fresh, C_fresh, PROBE, K), truth)

        occ = cnt[cnt > 0]
        results.append(
            dict(
                step=si + 1,
                quarter=qtr,
                n_new=int(len(new_idx)),
                n_total=int(len(cur)),
                dirty_leaves=int(obs),
                dirty_frac=obs / NL,
                null_dirty=float(null_d.mean()),
                null_sd=float(null_d.std()),
                iid_null_dirty=float(iid),
                dirty_nodes=int(dirty_nodes(new_leaves)),
                dirty_node_frac=dirty_nodes(new_leaves) / len(flat),
                recall_frozen=r_fro,
                recall_fresh=r_fresh,
                leaves_frozen=int(NL),
                leaves_fresh=int(NLf),
                max_leaf_frozen=int(occ.max()),
                p95_leaf_frozen=float(np.percentile(occ, 95)),
                fresh_build_s=build_s,
            )
        )
        z = (obs - null_d.mean()) / (null_d.std() + 1e-9)
        print(
            f"[{si + 1}/{len(steps)}] {qtr} +{len(new_idx):>5} (N={len(cur):>6})  "
            f"dirty {obs:>5}/{NL} ({100 * obs / NL:>4.1f}%)  "
            f"null {null_d.mean():>6.0f}+-{null_d.std():.0f} (z={z:>6.1f})  iid {iid:>6.0f}  |  "
            f"R@10 frozen {r_fro:.3f} fresh {r_fresh:.3f}",
            flush=True,
        )
        with open(os.path.join(S.CACHE, "seq_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nunseen-bucket fallbacks over full corpus: {unseen_all}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
