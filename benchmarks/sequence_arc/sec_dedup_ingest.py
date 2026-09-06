"""Dedup on ingest: what does removing 29% of the corpus actually buy?

`sec_dedup_ablation.py` found 29.1% of the SEC corpus is near-duplicate content. This asks
the operational question. OUTCOME VARIABLES, DECLARED BEFORE THE PROBE WAS WRITTEN (the
method lesson from this arc is that describing structure and then hunting for a use produced
five falsified hypotheses; the one result that got anywhere was framed against an outcome):

  1. PRIMARY: recall@10 against FULL-corpus exact kNN, at matched total work
  2. index size: points stored, leaves, nodes, bytes
  3. build time

HONEST EVALUATION. Deduping changes the corpus, so scoring against deduped ground truth
would answer an easier question. The retrieval task here is unchanged -- find neighbours in
the FULL corpus -- and the index simply stores one representative per duplicate cluster plus
a representative->members side table, expanding at query time. That is what a real
dedup-on-ingest pipeline does, and it keeps the comparison apples-to-apples.

  baseline        index all N points, score candidates directly
  dedup_inherit   index representatives only; retrieved representatives expand to their
                  cluster members, which INHERIT the representative's score. Members sit at
                  cos > 0.99 of the representative so the inherited score is accurate to
                  ~1e-2; expansion is an array lookup, so it costs no dot products. This is
                  where the win would come from.
  dedup_rescore   same, but members are actually scored. Costs the dot products back, so it
                  isolates how much of any gain is the cheap expansion versus a better tree.

DEDUP MUST BE TREE-FREE to count as "on ingest". Multi-table random-projection LSH:
sign bits over N_BITS random hyperplanes give buckets, near-duplicates are found by exact
comparison within buckets, and union-find merges pairs into clusters. A single table misses
pairs -- two vectors at cos 0.99 agree on one random bit with p = 1 - arccos(0.99)/pi
= 0.955, so over 12 bits p = 0.58 -- hence N_TABLES independent tables, giving
1 - 0.42^4 = 0.97 expected pair recall. The recovered rate is reported against the 29.1%
within-leaf figure as a cross-check.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_derived_bits import build, n_leaves  # noqa: E402
from sec_hier_routing import hier_probe  # noqa: E402

NQ = 1500
K = 10
MAX_PROBE = 192
PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 192)
DUP_COS = 0.99
N_TABLES = 4
N_BITS = 12
SEED = 42
BYTES_PER_VEC = 768 * 4


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def lsh_star_dedup(E, rng, n_tables=N_TABLES, n_bits=N_BITS, thresh=DUP_COS):
    """STAR clustering: a point joins a representative only if it is itself within `thresh`
    OF THAT REPRESENTATIVE. No transitivity.

    Transitive union-find (below) produced a 541-member "duplicate" cluster on this corpus,
    which is a chain A~B~C~...~Z where A and Z need not be similar at all. Members then
    inherit a representative score that is simply wrong for them. Star clustering bounds the
    error: every member is guaranteed within `thresh` of the vector representing it.
    """
    n = E.shape[0]
    assigned = np.full(n, -1, dtype=np.int64)
    for t in range(n_tables):
        H = rng.standard_normal((n_bits, E.shape[1])).astype(np.float32)
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        codes = ((E @ H.T > 0).astype(np.int64) << np.arange(n_bits)).sum(1)
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        bounds = np.append(np.searchsorted(cs, np.unique(cs)), len(cs))
        for bi in range(len(bounds) - 1):
            idx = order[bounds[bi] : bounds[bi + 1]]
            free = idx[assigned[idx] < 0]
            if len(free) < 2 or len(free) > 4000:
                continue
            G = E[free] @ E[free].T
            np.fill_diagonal(G, -1.0)
            taken = np.zeros(len(free), bool)
            for a in range(len(free)):
                if taken[a]:
                    continue
                grp = np.where((G[a] > thresh) & (~taken))[0]
                if len(grp) == 0:
                    continue
                taken[a] = True
                taken[grp] = True
                assigned[free[a]] = free[a]
                assigned[free[grp]] = free[a]
        done = (assigned >= 0).sum()
        print(f"    star table {t + 1}/{n_tables}: {done:,} points assigned to a cluster", flush=True)
    singles = assigned < 0
    assigned[singles] = np.where(singles)[0]
    return assigned, np.unique(assigned)


def lsh_dedup(E, rng, n_tables=N_TABLES, n_bits=N_BITS, thresh=DUP_COS):
    """Tree-free near-duplicate clustering. Returns (labels, reps) with labels[i] = cluster."""
    n, d = E.shape
    uf = UF(n)
    pairs = 0
    for t in range(n_tables):
        H = rng.standard_normal((n_bits, d)).astype(np.float32)
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        codes = ((E @ H.T > 0).astype(np.int64) << np.arange(n_bits)).sum(1)
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        bounds = np.searchsorted(cs, np.unique(cs))
        bounds = np.append(bounds, len(cs))
        for bi in range(len(bounds) - 1):
            idx = order[bounds[bi] : bounds[bi + 1]]
            if len(idx) < 2 or len(idx) > 4000:
                continue
            G = E[idx] @ E[idx].T
            np.fill_diagonal(G, -1.0)
            ii, jj = np.where(thresh < G)
            for a, b in zip(ii, jj):
                if a < b:
                    uf.union(int(idx[a]), int(idx[b]))
                    pairs += 1
        print(f"    table {t + 1}/{n_tables}: {pairs} cumulative dup pairs", flush=True)
    labels = np.array([uf.find(i) for i in range(n)])
    reps = np.unique(labels)
    return labels, reps


def build_expander(N, rep_members):
    """CSR-style expansion tables so cluster expansion is fully vectorised.

    The naive per-candidate python loop was the runtime bottleneck (~7.5M iterations at the
    top probe level). Returns (flat, start, count) indexed by GLOBAL point id.
    """
    start = np.zeros(N, dtype=np.int64)
    count = np.zeros(N, dtype=np.int64)
    chunks, pos = [], 0
    for g, grp in rep_members.items():
        start[g] = pos
        count[g] = len(grp)
        chunks.append(grp)
        pos += len(grp)
    return np.concatenate(chunks), start, count


def expand(cand, s, flat_m, start, count):
    """Vectorised: expand candidate reps to all cluster members, repeating their scores."""
    cnts = count[cand]
    tot = int(cnts.sum())
    if tot == 0:
        return cand, s
    st = np.repeat(start[cand], cnts)
    base = np.repeat(np.cumsum(cnts) - cnts, cnts)
    return flat_m[st + (np.arange(tot) - base)], np.repeat(s, cnts)


def evaluate(E, flat, Qe, truth, expander=None, score_members=False, probes=PROBES):
    """Recall@10 vs FULL-corpus truth. `expander` = (flat_members, start, count) or None."""
    NL = n_leaves(flat)
    mem = [None] * NL
    for nd in flat:
        if nd["leaf_id"] >= 0:
            mem[nd["leaf_id"]] = nd["indices"]
    sizes = np.array([len(m) if m is not None else 0 for m in mem])
    acc = {p: {"rec": [], "work": []} for p in probes}
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
                fm, st, ct = expander
                cand, s = expand(cand, s, fm, st, ct)
                if score_members:
                    s = E[cand] @ q  # pay the dot products back
                    work += len(cand)
            kk = min(K, len(cand))
            top = cand[np.argpartition(-s, kk - 1)[:kk]] if len(cand) > kk else cand
            acc[p]["rec"].append(len(set(top.tolist()) & set(truth[qi].tolist())) / K)
            acc[p]["work"].append(work)
    return [(float(np.mean(acc[p]["work"])), float(np.mean(acc[p]["rec"])), p) for p in probes if acc[p]["rec"]]


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    N = len(E)
    qi = rng.choice(N, NQ, replace=False)
    Qe = E[qi]
    print(f"corpus {E.shape}, {NQ} queries, truth = exact kNN over the FULL corpus", flush=True)
    truth = S.exact_knn(Qe, E, k=K)

    out, curves = {}, {}
    # ---- baseline ----
    t0 = time.time()
    flat_b = build(E, None, S.MAX_DEPTH, S.MIN_LEAF, "fixed", "origin", S.NUM_BITS, np.random.default_rng(SEED))
    bt = time.time() - t0
    curves["baseline"] = evaluate(E, flat_b, Qe, truth)
    out["baseline"] = {
        "points": N,
        "leaves": n_leaves(flat_b),
        "nodes": len(flat_b),
        "build_s": bt,
        "bytes": N * BYTES_PER_VEC,
    }
    print(f"\nbaseline      points={N:,} leaves={n_leaves(flat_b):,} build={bt:.0f}s", flush=True)

    # ---- two dedup clusterings: transitive union-find vs star ----------------------
    for kind, fn in [("transitive", lsh_dedup), ("star", lsh_star_dedup)]:
        print(f"\nLSH dedup, {kind} (tree-free, on-ingest):", flush=True)
        t0 = time.time()
        labels, reps = fn(E, np.random.default_rng(SEED))
        dedup_s = time.time() - t0
        removed = N - len(reps)
        grp = {}
        for g in range(N):
            grp.setdefault(int(labels[g]), []).append(g)
        rep_members = {k: np.array(v, dtype=np.int64) for k, v in grp.items()}
        csz = np.array([len(v) for v in rep_members.values()])
        print(
            f"  {len(reps):,} representatives, removed {removed:,} ({100 * removed / N:.1f}%) "
            f"[{dedup_s:.0f}s]; cluster max {csz.max()}, mean {csz.mean():.2f}, "
            f"singletons {(csz == 1).sum():,}",
            flush=True,
        )
        Er = np.ascontiguousarray(E[reps])
        t0 = time.time()
        flat_d = build(Er, None, S.MAX_DEPTH, S.MIN_LEAF, "fixed", "origin", S.NUM_BITS, np.random.default_rng(SEED))
        dt = time.time() - t0
        for nd in flat_d:
            nd["indices"] = reps[nd["indices"]]
        exp = build_expander(N, rep_members)
        for tag, rescore in [(f"{kind}_inherit", False), (f"{kind}_rescore", True)]:
            curves[tag] = evaluate(E, flat_d, Qe, truth, expander=exp, score_members=rescore)
        side = removed * 8
        out[f"dedup_{kind}"] = {
            "points": len(reps),
            "leaves": n_leaves(flat_d),
            "nodes": len(flat_d),
            "build_s": dt,
            "dedup_s": dedup_s,
            "bytes": len(reps) * BYTES_PER_VEC + side,
            "side_table_bytes": side,
            "cluster_max": int(csz.max()),
        }
        print(f"  indexed: points={len(reps):,} leaves={n_leaves(flat_d):,} build={dt:.0f}s", flush=True)

    print("\n" + "=" * 80)
    print("INDEX SIZE")
    print("=" * 80)
    b = out["baseline"]
    print(f"{'':<16}{'points':>12}{'leaves':>10}{'nodes':>10}{'vectors MB':>13}{'build s':>10}{'clust max':>11}")
    print(
        f"{'baseline':<16}{b['points']:>12,}{b['leaves']:>10,}{b['nodes']:>10,}"
        f"{b['bytes'] / 1e6:>13.1f}{b['build_s']:>10.0f}{'-':>11}"
    )
    for kind in ("transitive", "star"):
        dd = out.get(f"dedup_{kind}")
        if not dd:
            continue
        print(
            f"{'dedup_' + kind:<16}{dd['points']:>12,}{dd['leaves']:>10,}{dd['nodes']:>10,}"
            f"{dd['bytes'] / 1e6:>13.1f}{dd['build_s']:>10.0f}{dd['cluster_max']:>11}"
        )
        print(
            f"    reduction: points {100 * (1 - dd['points'] / b['points']):.1f}%  "
            f"leaves {100 * (1 - dd['leaves'] / b['leaves']):.1f}%  "
            f"vectors {100 * (1 - dd['bytes'] / b['bytes']):.1f}%  "
            f"(side table {dd['side_table_bytes'] / 1e6:.2f} MB, dedup {dd['dedup_s']:.0f}s)"
        )

    print("\n" + "=" * 80)
    print("RECALL@10 vs FULL-CORPUS TRUTH, at MATCHED TOTAL WORK")
    print("=" * 80)
    names = list(curves)

    def cv(c):
        pts = sorted((w, r) for w, r, _ in curves[c])
        return np.array([a for a, _ in pts]), np.array([bb for _, bb in pts])

    lo = max(cv(c)[0].min() for c in names)
    hi = min(cv(c)[0].max() for c in names)
    budgets = np.unique(np.round(np.geomspace(lo, hi, 6)).astype(int))
    print(f"{'condition':<16}" + "".join(f"{x:>11}" for x in budgets))
    vals = {}
    for c in names:
        x, y = cv(c)
        vals[c] = [float(np.interp(np.log(t), np.log(x), y)) for t in budgets]
        print(f"{c:<16}" + "".join(f"{v:>11.4f}" for v in vals[c]))
    print("\ndelta vs baseline:")
    for c in names[1:]:
        print(f"  {c:<16}" + "".join(f"{vals[c][i] - vals['baseline'][i]:>+11.4f}" for i in range(len(budgets))))

    print("\nwork to reach a target recall (lower is better):")
    for tr in (0.80, 0.90, 0.95):
        row = f"  recall {tr:.2f}: "
        for c in names:
            x, y = cv(c)
            o = np.argsort(y)
            w = float(np.exp(np.interp(tr, y[o], np.log(x[o])))) if y.min() <= tr <= y.max() else float("nan")
            row += f"{c}={w:.0f} " if np.isfinite(w) else f"{c}=n/a "
        print(row)

    path = os.path.join(S.CACHE, "dedup_ingest_results.json")
    with open(path, "w") as f:
        json.dump(
            {
                "size": out,
                "curves": {k: [{"work": w, "recall": r, "probe": p} for w, r, p in v] for k, v in curves.items()},
            },
            f,
            indent=2,
        )
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
