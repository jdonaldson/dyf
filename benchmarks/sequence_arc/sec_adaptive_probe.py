"""THE LEVER TEST: does eff_rank-weighted probe allocation actually buy anything?

sec_cell_spectra.py found per-cell effective rank predicts retrieval difficulty where
occupancy completely fails (+0.314 raw, +0.405 partialling log n, vs log_n at -0.002,
null95 0.186). A correlation with difficulty is NOT a lever. This spends a FIXED total
work budget non-uniformly across queries -- more probes for queries landing in
high-eff_rank cells, fewer for boilerplate -- and asks whether recall improves at matched
total work. This is the cheapest kill for the whole spectral direction: SEC is where the
correlation is strongest, so failure here closes it everywhere.

CONDITIONS. The competitor is not "nothing". dyf ALREADY ships query-adaptive probing keyed
on routing margin (`_resolve_nprobe` / `AdaptiveProbeConfig`, lazy_index.py:2136), so that
is the incumbent to beat, and cell size is the free baseline that must NOT work:

  uniform    every query gets the same probe count                     (baseline)
  eff_rank   more probes where the routed cell has high effective rank (the proposal)
  log_n      more probes where the routed cell is large   (negative control: rho was -0.002)
  margin     more probes where the primary path ran close to a boundary (the INCUMBENT)
  combo      eff_rank + margin, averaged percentiles

ALLOCATION. Rank-based so no signal's scale matters: a query at percentile u of its signal
gets base * (1 + ALPHA * (2u - 1)) probes, clipped to [1, MAX_PROBE]. ALPHA=0 is uniform.

MATCHED WORK, NOT MATCHED PROBES. Leaves differ in size, so giving a probe to a query in a
big-leaf region costs more than in a small-leaf region -- equal mean probe count is NOT
equal work. Each condition sweeps `base` to trace a work/recall curve and the curves are
compared at interpolated equal total work (routing dots + members scanned).

ONE TRAVERSAL PER QUERY. hier_probe is run once at MAX_PROBE, recording the leaf collection
order and cumulative routing cost; every condition and every budget is a different PREFIX of
that same order, so allocation policy costs nothing extra to evaluate.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_spectra import descriptors, spectrum  # noqa: E402
from sec_cell_volume import cells_at_depth, flatten_ev  # noqa: E402

NQ = 1500
K = 10
MAX_PROBE = 192
BASES = (4, 8, 16, 32, 64, 128)
ALPHA = 0.6  # allocation spread: 0.4x .. 1.6x of base
N_SUB = 200
CELL_DEPTH = 2
SEED = 42


def hier_probe_full(q, flat, max_probe=MAX_PROBE):
    """Descend like dyf, returning leaf order, cumulative routing dots, and primary margin."""
    import heapq

    order, costs = [], []
    dots, ctr = 0, 0
    min_margin = np.inf
    heap = [(0.0, 0, 0, True)]  # (priority, tiebreak, node, on_primary_path)
    seen = set()
    while heap and len(order) < max_probe:
        pri, _, nid, primary = heapq.heappop(heap)
        nd = flat[nid]
        if nd["leaf_id"] >= 0:
            if nd["leaf_id"] not in seen:
                seen.add(nd["leaf_id"])
                order.append(nd["leaf_id"])
                costs.append(dots)
            continue
        H, bmap, kids = nd["hp"], nd["bmap"], nd["children"]
        if H is None:
            ctr += 1
            heapq.heappush(heap, (pri, ctr, kids[0], primary))
            continue
        proj = H @ q
        dots += H.shape[0]
        absp = np.abs(proj)
        if primary:
            min_margin = min(min_margin, float(absp.min()))
        bid = int(((proj > 0).astype(np.int64) << np.arange(H.shape[0])).sum())
        if bid in bmap:
            pbid = bid
        else:
            pbid = min(bmap, key=lambda b: absp[[i for i in range(H.shape[0]) if (bid ^ b) & (1 << i)]].sum())
        ctr += 1
        heapq.heappush(heap, (pri, ctr, kids[bmap[pbid]], primary))
        for b, ci in bmap.items():
            if b == pbid:
                continue
            d = float(absp[[i for i in range(H.shape[0]) if (bid ^ b) & (1 << i)]].sum())
            ctr += 1
            heapq.heappush(heap, (pri + d, ctr, kids[ci], False))
    return order, costs, (0.0 if not np.isfinite(min_margin) else min_margin)


def pct_rank(v):
    v = np.asarray(v, float)
    r = np.argsort(np.argsort(v)).astype(float)
    return r / max(len(v) - 1, 1)


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    t0 = time.time()
    flat = flatten_ev(
        build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), E
    )
    NL = S.n_leaves(flat)
    print(f"tree: {len(flat)} nodes, {NL} leaves [{time.time() - t0:.0f}s]", flush=True)

    # per-cell eff_rank at matched n, mapped to every leaf
    leaf_eff = np.full(NL, np.nan)
    leaf_logn = np.full(NL, np.nan)
    cells = cells_at_depth(flat, CELL_DEPTH)
    for nid, lids in cells:
        idx = flat[nid]["indices"]
        p = spectrum(E[idx], N_SUB, rng, 5)
        er = descriptors(p)["eff_rank"] if p is not None else np.nan
        for lf in lids:
            leaf_eff[lf] = er
            leaf_logn[lf] = np.log(max(len(idx), 1))
    ok = np.isfinite(leaf_eff)
    leaf_eff[~ok] = np.nanmedian(leaf_eff)
    leaf_logn[~np.isfinite(leaf_logn)] = np.nanmedian(leaf_logn)
    print(f"eff_rank mapped for {ok.sum()}/{NL} leaves ({len(cells)} depth-{CELL_DEPTH} cells)", flush=True)

    qi = rng.choice(len(E), NQ, replace=False)
    Qe = E[qi]
    truth = S.exact_knn(Qe, E, k=K)

    mem = [None] * NL
    for nd in flat:
        if nd["leaf_id"] >= 0:
            mem[nd["leaf_id"]] = nd["indices"]
    sizes = np.array([len(m) if m is not None else 0 for m in mem])

    print("traversing...", flush=True)
    t0 = time.time()
    trav = [hier_probe_full(Qe[i], flat) for i in range(NQ)]
    print(f"  {NQ} traversals [{time.time() - t0:.0f}s]", flush=True)

    # per-query signals, from the FIRST leaf the query lands in (available at query time)
    first_leaf = np.array([o[0] if o else 0 for o, _, _ in trav])
    sig = {
        "uniform": np.zeros(NQ),
        "eff_rank": leaf_eff[first_leaf],
        "log_n": leaf_logn[first_leaf],
        "margin": -np.array([m for _, _, m in trav]),  # small margin -> more probes
    }
    sig["combo"] = (pct_rank(sig["eff_rank"]) + pct_rank(sig["margin"])) / 2
    pcts = {k: (np.full(NQ, 0.5) if k == "uniform" else pct_rank(v)) for k, v in sig.items()}

    def evaluate(name, base):
        alloc = np.clip(np.round(base * (1 + ALPHA * (2 * pcts[name] - 1))).astype(int), 1, MAX_PROBE)
        recs, works = [], []
        for i in range(NQ):
            order, rcost, _ = trav[i]
            if not order:
                continue
            m = int(min(alloc[i], len(order)))
            cand = np.concatenate([mem[lf] for lf in order[:m]])
            s = E[cand] @ Qe[i]
            kk = min(K, len(cand))
            top = cand[np.argpartition(-s, kk - 1)[:kk]] if len(cand) > kk else cand
            recs.append(len(set(top.tolist()) & set(truth[i].tolist())) / K)
            works.append(rcost[m - 1] + int(sizes[np.array(order[:m])].sum()))
        return float(np.mean(works)), float(np.mean(recs))

    conds = ["uniform", "eff_rank", "log_n", "margin", "combo"]
    curves = {c: [] for c in conds}
    for c in conds:
        for b in BASES:
            w, r = evaluate(c, b)
            curves[c].append({"base": b, "work": w, "recall": r})
        print(f"  {c:<10} " + "  ".join(f"{p['recall']:.3f}@{p['work']:.0f}" for p in curves[c]), flush=True)

    print("\n" + "=" * 80)
    print(f"Recall@10 at MATCHED TOTAL WORK (alpha={ALPHA}; allocation reshuffles a fixed budget)")
    print("=" * 80)
    lo = max(min(p["work"] for p in curves[c]) for c in conds)
    hi = min(max(p["work"] for p in curves[c]) for c in conds)
    budgets = np.unique(np.round(np.geomspace(lo, hi, 6)).astype(int))

    def interp(c, b):
        pts = sorted((p["work"], p["recall"]) for p in curves[c])
        x = np.log([p[0] for p in pts])
        y = [p[1] for p in pts]
        return float(np.interp(np.log(b), x, y))

    print(f"{'condition':<12}" + "".join(f"{b:>11}" for b in budgets))
    vals = {c: [interp(c, b) for b in budgets] for c in conds}
    for c in conds:
        print(f"{c:<12}" + "".join(f"{v:>11.4f}" for v in vals[c]))
    print("\ndelta vs uniform:")
    for c in conds[1:]:
        print(f"  {c:<12}" + "".join(f"{vals[c][i] - vals['uniform'][i]:>+11.4f}" for i in range(len(budgets))))
    print("\ndelta vs margin (the incumbent adaptive scheme):")
    for c in ["eff_rank", "combo"]:
        print(f"  {c:<12}" + "".join(f"{vals[c][i] - vals['margin'][i]:>+11.4f}" for i in range(len(budgets))))

    path = os.path.join(S.CACHE, "adaptive_probe_results.json")
    with open(path, "w") as f:
        json.dump({"budgets": budgets.tolist(), "curves": curves, "alpha": ALPHA}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
