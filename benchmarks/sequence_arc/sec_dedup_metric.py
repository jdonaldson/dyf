"""Was the dedup recall penalty an artifact of the metric? Distinct-content recall.

sec_dedup_ingest.py scored recall@10 against exact kNN over the raw corpus and found dedup
losing up to 5pp, with a clean dose-response in duplicate-cluster size. But that metric asks
"did you reproduce the brute-force list", and when a corpus holds 4 near-identical copies of
a document the true top-10 spends 4 of its 10 slots on the SAME CONTENT. Raw recall@10
therefore REWARDS returning duplicates -- exactly what dedup exists to prevent. The penalty
measured there may be the metric, not the system.

TWO METRICS, SAME RUNS:

  raw@10       |retrieved_top10 INTERSECT true_top10| / 10, truth = exact kNN over the full
               corpus. Right question if you are benchmarking an ANN index as an
               approximation to brute force. Duplicate-rewarding.
  distinct@10  Ground truth = the top-10 CLUSTERS ranked by their best member's score.
               Prediction = the system's returned items mapped to clusters, deduplicated in
               rank order, first 10 distinct. Right question if you are serving users, who
               do not want four copies of one document in ten results.

FAIRNESS. distinct@10 applies the same scoring convention to BOTH arms -- the baseline is
also credited only for distinct content, so it is penalised for spending result slots on
duplicates. That is the point: it is a different question, not a handicap. Cluster labels are
a scoring artefact here, available to the grader rather than to either index.

Neither metric is wrong. They answer different questions, and the dedup decision hinges on
which one is intended.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_dedup_ingest import MAX_PROBE, SEED, K, build_expander, expand, lsh_star_dedup  # noqa: E402
from sec_derived_bits import build, n_leaves  # noqa: E402
from sec_hier_routing import hier_probe  # noqa: E402

NQ = 1200
PROBES = (2, 4, 8, 16, 32, 64, 128, 192)


def cluster_truth(E, Qe, labels_compact, n_clusters, k=K, chunk=8192):
    """Top-k CLUSTERS per query, ranked by each cluster's best-scoring member."""
    order = np.argsort(labels_compact, kind="stable")
    lab_sorted = labels_compact[order]
    starts = np.searchsorted(lab_sorted, np.arange(n_clusters))
    out = np.zeros((len(Qe), k), dtype=np.int64)
    for qi in range(len(Qe)):
        s = np.empty(len(E), np.float32)
        for c in range(0, len(E), chunk):
            s[c : c + chunk] = E[c : c + chunk] @ Qe[qi]
        cmax = np.maximum.reduceat(s[order], starts)
        top = np.argpartition(-cmax, k - 1)[:k]
        out[qi] = top[np.argsort(-cmax[top])]
    return out


def first_n_distinct(items, labels_compact, s, n=K):
    """Map items to clusters in descending-score order, keep first n distinct clusters."""
    o = np.argsort(-s)
    seen, res = set(), []
    for i in o:
        c = int(labels_compact[items[i]])
        if c not in seen:
            seen.add(c)
            res.append(c)
            if len(res) == n:
                break
    return res


def run(E, flat, Qe, truth_raw, truth_cl, labels_compact, expander=None, probes=PROBES):
    NL = n_leaves(flat)
    mem = [None] * NL
    for nd in flat:
        if nd["leaf_id"] >= 0:
            mem[nd["leaf_id"]] = nd["indices"]
    sizes = np.array([len(m) if m is not None else 0 for m in mem])
    acc = {p: {"raw": [], "dist": [], "work": []} for p in probes}
    for qi in range(len(Qe)):
        q = Qe[qi]
        order, rcost = hier_probe(q, flat, MAX_PROBE)
        if not order:
            continue
        cum = np.cumsum(sizes[np.array(order)])
        for p in probes:
            m = min(p, len(order))
            cand = np.concatenate([mem[lf] for lf in order[:m]])
            s = E[cand] @ q
            work = rcost[m - 1] + int(cum[m - 1])
            if expander is not None:
                cand, s = expand(cand, s, *expander)
            kk = min(K, len(cand))
            top = cand[np.argpartition(-s, kk - 1)[:kk]] if len(cand) > kk else cand
            acc[p]["raw"].append(len(set(top.tolist()) & set(truth_raw[qi].tolist())) / K)
            pred_cl = first_n_distinct(cand, labels_compact, s, K)
            acc[p]["dist"].append(len(set(pred_cl) & set(truth_cl[qi].tolist())) / K)
            acc[p]["work"].append(work)
    return [
        (float(np.mean(acc[p]["work"])), float(np.mean(acc[p]["raw"])), float(np.mean(acc[p]["dist"])), p)
        for p in probes
        if acc[p]["raw"]
    ]


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    N = len(E)
    qi = rng.choice(N, NQ, replace=False)
    Qe = E[qi]
    print(f"corpus {E.shape}, {NQ} queries", flush=True)

    labels, reps = lsh_star_dedup(E, np.random.default_rng(SEED))
    remap = {int(r): i for i, r in enumerate(reps)}
    labels_compact = np.array([remap[int(x)] for x in labels], dtype=np.int64)
    C = len(reps)
    print(f"star dedup: {C:,} clusters ({100 * (1 - C / N):.1f}% collapsed)", flush=True)

    t0 = time.time()
    truth_raw = S.exact_knn(Qe, E, k=K)
    truth_cl = cluster_truth(E, Qe, labels_compact, C)
    print(f"ground truth: raw + cluster-level [{time.time() - t0:.0f}s]", flush=True)

    dup_in_truth = np.array([K - len(set(labels_compact[truth_raw[i]].tolist())) for i in range(NQ)])
    print(
        f"duplicate slots in the raw top-10: mean {dup_in_truth.mean():.2f} of 10, "
        f"{100 * (dup_in_truth > 0).mean():.0f}% of queries have at least one",
        flush=True,
    )

    grp = {}
    for g in range(N):
        grp.setdefault(int(labels[g]), []).append(g)
    rep_members = {k: np.array(v, dtype=np.int64) for k, v in grp.items()}

    flat_b = build(E, None, S.MAX_DEPTH, S.MIN_LEAF, "fixed", "origin", S.NUM_BITS, np.random.default_rng(SEED))
    Er = np.ascontiguousarray(E[reps])
    flat_d = build(Er, None, S.MAX_DEPTH, S.MIN_LEAF, "fixed", "origin", S.NUM_BITS, np.random.default_rng(SEED))
    for nd in flat_d:
        nd["indices"] = reps[nd["indices"]]
    exp = build_expander(N, rep_members)

    curves = {
        "baseline": run(E, flat_b, Qe, truth_raw, truth_cl, labels_compact),
        "dedup": run(E, flat_d, Qe, truth_raw, truth_cl, labels_compact, expander=exp),
    }

    print("\n" + "=" * 78)
    print("SAME RUNS, TWO METRICS, at matched total work")
    print("=" * 78)

    def cv(c, mi):
        pts = sorted((r[0], r[mi]) for r in curves[c])
        return np.array([a for a, _ in pts]), np.array([b for _, b in pts])

    lo = max(cv(c, 1)[0].min() for c in curves)
    hi = min(cv(c, 1)[0].max() for c in curves)
    budgets = np.unique(np.round(np.geomspace(lo, hi, 6)).astype(int))
    for mi, lab in [(1, "raw@10 (duplicate-rewarding)"), (2, "distinct@10 (content-level)")]:
        print(f"\n{lab}")
        print(f"{'condition':<12}" + "".join(f"{b:>11}" for b in budgets))
        vals = {}
        for c in curves:
            x, y = cv(c, mi)
            vals[c] = [float(np.interp(np.log(b), np.log(x), y)) for b in budgets]
            print(f"{c:<12}" + "".join(f"{v:>11.4f}" for v in vals[c]))
        print(
            f"{'delta':<12}" + "".join(f"{vals['dedup'][i] - vals['baseline'][i]:>+11.4f}" for i in range(len(budgets)))
        )

    path = os.path.join(S.CACHE, "dedup_metric_results.json")
    with open(path, "w") as f:
        json.dump(
            {
                "budgets": budgets.tolist(),
                "curves": {
                    k: [{"work": w, "raw": r, "distinct": d, "probe": p} for w, r, d, p in v] for k, v in curves.items()
                },
                "dup_slots_mean": float(dup_in_truth.mean()),
            },
            f,
            indent=2,
        )
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
