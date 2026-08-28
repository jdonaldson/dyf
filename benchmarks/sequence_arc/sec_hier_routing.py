"""Does the derived-bits vs median-offset ranking survive dyf's REAL router?

sec_derived_bits.py compared trees through sec_seqlib.ivf_search -- flat IVF, which scans
EVERY leaf centroid to pick probes. dyf does not do that. `LazyIndex._find_candidate_leaves`
(src/dyf/lazy_index.py:2136) descends the tree from the root, and it never touches a leaf
centroid: it hashes the query against each node's hyperplanes and explores alternative
buckets in order of MARGIN DISTANCE,

    cost of flipping bit i = |projection[i]|          (lazy_index.py:1590)

so the two routers differ in signal, not just in cost. Two consequences:

  COST     hierarchical routing costs sum(num_bits) over visited internal nodes, not
           n_leaves. Leaf count stops being a first-order search cost, which is exactly
           the control sec_derived_bits.py had to spend its degrees of freedom on.
  QUALITY  the margin is measured from the ORIGIN. On an 80/20 bit the mean projection
           sits ~0.8 sd from zero, so |projection| overstates the cost of flipping that
           bit and the alternative ordering is miscalibrated. Centring the cut should
           help MORE here than under flat IVF, where the offset only affected which
           points shared a leaf.

Prediction being tested: median offset gains relative to derived bits when the router is
the real one. Falsifiable -- the ranking may simply hold.

Efficiency: one traversal per (query, tree) at max nprobe records the leaf collection
ORDER and the cumulative dot-product cost at each collection. Every smaller nprobe is a
prefix of that, so all probe budgets come from a single pass.
"""

import heapq
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_derived_bits import MATCHED_MIN_LEAF, build, n_leaves  # noqa: E402

NQ = 800
K = 10
MAX_PROBE = 256
PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEED = 42


def hier_probe(q, flat, max_probe=MAX_PROBE):
    """dyf's traversal. Returns (leaf_ids in collection order, cumulative routing dots).

    Faithful to _find_candidate_leaves: priority queue over (margin cost, node), primary
    child at the parent's own priority, alternatives at priority + margin_distance.
    Offsets are honoured, so the margin is |proj - off| -- the distance to the ACTUAL cut.
    """
    order, costs = [], []
    dots = 0
    ctr = 0
    heap = [(0.0, 0, 0)]  # (priority, tiebreak, node_id)
    seen = set()
    while heap and len(order) < max_probe:
        pri, _, nid = heapq.heappop(heap)
        nd = flat[nid]
        if nd["leaf_id"] >= 0:
            if nd["leaf_id"] not in seen:
                seen.add(nd["leaf_id"])
                order.append(nd["leaf_id"])
                costs.append(dots)
            continue
        H, off, bmap, kids = nd["H"], nd["off"], nd["bmap"], nd["children"]
        if H is None:
            ctr += 1
            heapq.heappush(heap, (pri, ctr, kids[0]))
            continue
        proj = H @ q - off  # signed distance to the actual cut
        dots += H.shape[0]
        bits = (proj > 0).astype(np.int64)
        bid = int((bits << np.arange(H.shape[0])).sum())
        absp = np.abs(proj)
        # primary: exact bucket if present, else nearest by margin
        if bid in bmap:
            primary_bid = bid
        else:
            primary_bid = min(bmap, key=lambda b: absp[_flipped(bid, b, H.shape[0])].sum())
        ctr += 1
        heapq.heappush(heap, (pri, ctr, kids[bmap[primary_bid]]))
        for b, ci in bmap.items():
            if b == primary_bid:
                continue
            d = float(absp[_flipped(bid, b, H.shape[0])].sum())
            ctr += 1
            heapq.heappush(heap, (pri + d, ctr, kids[ci]))
    return order, costs


def _flipped(a, b, nb):
    """Indices of bits differing between a and b."""
    x = a ^ b
    return [i for i in range(nb) if x & (1 << i)]


def leaf_members(flat, NL):
    mem = [None] * NL
    for nd in flat:
        if nd["leaf_id"] >= 0:
            mem[nd["leaf_id"]] = nd["indices"]
    return mem


def evaluate_hier(E, flat, Qe, truth, probes=PROBES):
    NL = n_leaves(flat)
    mem = leaf_members(flat, NL)
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
            kk = min(K, len(cand))
            top = cand[np.argpartition(-s, kk - 1)[:kk]] if len(cand) > kk else cand
            acc[p]["rec"].append(len(set(top.tolist()) & set(truth[qi].tolist())) / K)
            # work = routing dots up to the m-th leaf + members scanned
            acc[p]["work"].append(rcost[m - 1] + int(cum[m - 1]))
    return [(float(np.mean(acc[p]["work"])), float(np.mean(acc[p]["rec"])), p) for p in probes if acc[p]["rec"]]


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(E), NQ, replace=False)
    Qe = E[qi]
    print(f"corpus {E.shape}, {NQ} queries, hierarchical routing (dyf's own)", flush=True)
    truth = S.exact_knn(Qe, E, k=K)

    conds = [
        ("fixed4_origin", "fixed", "origin"),
        ("fixed4_median", "fixed", "median"),
        ("derived_origin", "derived", "origin"),
        ("derived_median", "derived", "median"),
    ]
    out = {}
    for name, bm, om in conds:
        ml = MATCHED_MIN_LEAF[name]
        t0 = time.time()
        flat = build(E, None, S.MAX_DEPTH, ml, bm, om, S.NUM_BITS, np.random.default_rng(SEED))
        pts = evaluate_hier(E, flat, Qe, truth)
        out[name] = {
            "leaves": n_leaves(flat),
            "min_leaf": ml,
            "curve": [{"work": w, "recall": r, "probe": p} for w, r, p in pts],
        }
        print(
            f"{name:<16} leaves={n_leaves(flat):>6}  "
            + "  ".join(f"p{p}:{r:.3f}@{w:.0f}" for w, r, p in pts[:4])
            + f"  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    print("\n" + "=" * 80)
    print("Recall@10 vs TOTAL WORK (routing dots + members scanned), hierarchical router")
    print("=" * 80)

    def curve(c):
        pts = sorted((p["work"], p["recall"]) for p in out[c]["curve"])
        return np.array([a for a, _ in pts]), np.array([b for _, b in pts])

    lo = max(curve(c)[0].min() for c, _, _ in conds)
    hi = min(curve(c)[0].max() for c, _, _ in conds)
    budgets = np.unique(np.round(np.geomspace(lo, hi, 6)).astype(int))
    print(f"{'condition':<16}" + "".join(f"{b:>11}" for b in budgets))
    vals = {}
    for c, _, _ in conds:
        x, y = curve(c)
        vals[c] = [float(np.interp(np.log(b), np.log(x), y)) for b in budgets]
        print(f"{c:<16}" + "".join(f"{v:>11.4f}" for v in vals[c]))
    print("\ndelta vs fixed4_origin:")
    for c, _, _ in conds[1:]:
        print(f"  {c:<16}" + "".join(f"{vals[c][i] - vals['fixed4_origin'][i]:>+11.4f}" for i in range(len(budgets))))

    path = os.path.join(S.CACHE, "hier_routing_results.json")
    with open(path, "w") as f:
        json.dump({"budgets": budgets.tolist(), "results": out}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
