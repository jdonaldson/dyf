"""Can num_bits be DERIVED per node instead of hand-set, and does centring the cut help?

Two independent changes to how a dyf node splits, measured factorially:

  BITS    fixed    num_bits = 4 everywhere (current default)
          derived  b = min(significant PCs, capacity cap, B_MAX)
  OFFSET  origin   cut at `x @ h > 0`          (current behaviour)
          median   cut at `x @ h > median(proj)` (centre of mass on that axis)

Why derived bits. `sec_split_anatomy.py` established that a node's hyperplanes ARE its
top-num_bits PCs, so num_bits is literally "how many eigenvectors do we trust". That is
an estimable quantity, not a taste parameter. Two limits bound it:

  signal    how many PCs clear a sampling-noise floor. Estimated by HORN'S PARALLEL
            ANALYSIS: shuffle each column independently -- which destroys cross-column
            correlation while preserving every marginal -- recompute the spectrum, and
            keep components above the permuted 95th percentile. Preferred over a
            Marchenko-Pastur edge because MP assumes white noise and embedding noise is
            demonstrably coloured.
  capacity  a node splitting into 2^b buckets leaves n/2^b points each; below the leaf
            floor the extra bits only manufacture undersized buckets.
            b <= log2(n / (2 * min_leaf)).

Why the offset. Routing is `x @ H.T > 0` while the PCs are fitted on CENTRED data, so
nothing makes the cut pass through the cell. Measured per-bit frac>0 on a 20k subset:
0.585 / 0.118 / 0.219 / 0.239 -- bits 2-4 are 80/20 slabs. Effective buckets 7.8 of 16.

FAIRNESS. Every condition is built and searched by the SAME numpy harness, and recall is
compared at MATCHED SCAN COST (mean candidates examined), not matched probe count --
these trees have different leaf-size distributions, so equal probe is not equal work.
`--validate` checks the fixed+origin arm reproduces dyf's own build_dyf_tree.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

B_MAX = 6
B_MIN = 1
N_PERM = 3  # parallel-analysis permutations per node
PA_SUBSAMPLE = 1500  # cap points used for the PA test
NQ = 1500
K = 10
PROBES = (4, 8, 16, 32, 64, 128, 256)
SEED = 42


# --------------------------------------------------------------------------- components
def top_eigs(Z, k, seed=0):
    """Top-k singular values of centred Z, via randomized SVD."""
    from sklearn.utils.extmath import randomized_svd

    k = int(min(k, min(Z.shape) - 1))
    if k < 1:
        return np.zeros(0), np.zeros((0, Z.shape[1]))
    _, s, Vt = randomized_svd(Z, n_components=k, random_state=seed, n_iter=4)
    return s, Vt


def n_significant(Z, rng, b_max=B_MAX, n_perm=N_PERM):
    """Horn's parallel analysis: how many LEADING components beat column-shuffled data?

    Column-wise shuffling destroys cross-column correlation while preserving every
    marginal, so the permuted spectrum is the noise floor for this exact data shape.
    Counts a PREFIX (stops at the first component that fails) since num_bits selects the
    top-b eigenvectors -- a non-contiguous "significant" set is not usable as a count.

    Validated (scratchpad, 2026-08-28): returns 6 (=B_MAX) on corpus-scale subsets where
    var shares are 0.126/0.067/0.051/0.038 vs an isotropic 0.0013, and exactly 2 on a
    synthetic 3-cluster control. Ratios observed/shuffled at n=1500: 5.4/4.0/3.5/3.1.
    """
    n = len(Z)
    if n > PA_SUBSAMPLE:
        Z = Z[rng.choice(n, PA_SUBSAMPLE, replace=False)]
    Z = np.ascontiguousarray(Z - Z.mean(0))
    k = int(min(b_max, min(Z.shape) - 1))
    if k < 1:
        return 0
    s_obs, _ = top_eigs(Z, k, seed=0)
    null = np.zeros((n_perm, k))
    for r in range(n_perm):
        # vectorised independent per-column shuffle (the python column loop dominated)
        order = np.argsort(rng.random(Z.shape), axis=0)
        P = np.take_along_axis(Z, order, axis=0)
        s_null, _ = top_eigs(P, k, seed=r + 1)
        null[r, : len(s_null)] = s_null
    thresh = np.percentile(null, 95, axis=0)
    passed = s_obs[: len(thresh)] > thresh
    return int(np.argmin(passed)) if not passed.all() else int(len(passed))


def choose_bits(Z, n, min_leaf, rng, mode, fixed_bits):
    if mode == "fixed":
        return fixed_bits
    cap = int(np.floor(np.log2(max(n / (2.0 * min_leaf), 2.0))))
    sig = n_significant(Z, rng)
    return int(np.clip(min(sig, cap), B_MIN, B_MAX))


# ------------------------------------------------------------------------- tree building
def build(E, idx, max_depth, min_leaf, bits_mode, offset_mode, fixed_bits, rng, seed=SEED):
    """Numpy dyf-style tree. Nodes carry (H, offsets, bmap) so routing is a sign test."""
    nodes = []

    def rec(ids, depth):
        nid = len(nodes)
        nodes.append(None)
        node = {"children": [], "H": None, "off": None, "bmap": None, "indices": ids, "leaf_id": -1}
        nodes[nid] = node
        if depth == 0 or len(ids) < 2 * min_leaf:
            return nid
        Z = E[ids].astype(np.float32)
        Zc = Z - Z.mean(0)
        b = choose_bits(Zc, len(ids), min_leaf, rng, bits_mode, fixed_bits)
        if b < 1:
            return nid
        _, Vt = top_eigs(Zc, b, seed=seed)
        if len(Vt) == 0:
            return nid
        H = np.ascontiguousarray(Vt[: len(Vt)], dtype=np.float32)
        proj = Z @ H.T
        off = (
            np.zeros(H.shape[0], np.float32) if offset_mode == "origin" else np.median(proj, axis=0).astype(np.float32)
        )
        bits = (proj > off).astype(np.int64)
        bid = (bits << np.arange(H.shape[0])).sum(1)
        uniq = np.unique(bid)
        if len(uniq) <= 1:
            return nid
        node["H"], node["off"] = H, off
        node["bmap"] = {int(u): i for i, u in enumerate(uniq)}
        kids = []
        for u in uniq:
            sub = ids[bid == u]
            kids.append(rec(sub, depth - 1) if len(sub) >= 2 * min_leaf else rec(sub, 0))
        node["children"] = kids
        return nid

    rec(np.arange(len(E), dtype=np.int64) if idx is None else idx, max_depth)
    lid = 0
    for nd in nodes:
        if not nd["children"]:
            nd["leaf_id"] = lid
            lid += 1
    return nodes


def n_leaves(flat):
    return sum(1 for n in flat if n["leaf_id"] >= 0)


def assign_from_build(flat, n):
    a = np.full(n, -1, np.int32)
    for nd in flat:
        if nd["leaf_id"] >= 0:
            a[nd["indices"]] = nd["leaf_id"]
    return a


def route(E, flat):
    """Route arbitrary points through the frozen structure (same fallback as seqlib)."""
    out = np.full(len(E), -1, np.int32)
    cents = {}

    def centroid(nid):
        if nid not in cents:
            ids = flat[nid]["indices"]
            c = E[ids].mean(0) if len(ids) else np.zeros(E.shape[1], np.float32)
            cents[nid] = c / (np.linalg.norm(c) + 1e-12)
        return cents[nid]

    stack = [(0, np.arange(len(E), dtype=np.int64))]
    while stack:
        nid, ids = stack.pop()
        nd = flat[nid]
        if not len(ids):
            continue
        if nd["leaf_id"] >= 0:
            out[ids] = nd["leaf_id"]
            continue
        H, off, bmap, kids = nd["H"], nd["off"], nd["bmap"], nd["children"]
        if H is None:
            stack.append((kids[0], ids))
            continue
        bid = ((E[ids] @ H.T > off).astype(np.int64) << np.arange(H.shape[0])).sum(1)
        lut = np.full(1 << H.shape[0], -1, np.int64)
        for bb, c in bmap.items():
            lut[bb] = c
        child = lut[bid]
        miss = child < 0
        if miss.any():
            kc = np.stack([centroid(k) for k in kids])
            child[miss] = (E[ids[miss]] @ kc.T).argmax(1)
        for ci, k in enumerate(kids):
            sel = child == ci
            if sel.any():
                stack.append((k, ids[sel]))
    return out


def balance_stats(flat):
    """Mean effective-bucket count over internal nodes, and per-bit balance."""
    eb, bl, bits = [], [], []
    for nd in flat:
        if nd["H"] is None:
            continue
        kids = nd["children"]
        cnt = np.array([len(flat[k]["indices"]) for k in kids], float)
        if cnt.sum() <= 0:
            continue
        sh = cnt / cnt.sum()
        eb.append(float(np.exp(-(sh * np.log(np.clip(sh, 1e-12, None))).sum())))
        bits.append(nd["H"].shape[0])
        Z = nd["indices"]
        proj = None
        bl.append(float(1.0 - abs(2.0 * (sh.max()) - 1.0)) if len(sh) else np.nan)
        del Z, proj
    return {
        "mean_eff_buckets": float(np.mean(eb)) if eb else np.nan,
        "mean_bits": float(np.mean(bits)) if bits else np.nan,
        "internal_nodes": len(eb),
        "bits_hist": np.bincount(bits, minlength=B_MAX + 1).tolist() if bits else [],
    }


# --------------------------------------------------------------------------------- eval
def evaluate(E, flat, Q, truth, probes=PROBES):
    a = assign_from_build(flat, len(E))
    NL = n_leaves(flat)
    C, _ = S.leaf_centroids(E, a, NL)
    pts = []
    for p in probes:
        rec = S.recall_at_k(S.ivf_search(Q, E, a, C, p, K), truth)
        cost = S.scan_cost(Q, a, C, p)
        pts.append((cost, rec, p))
    return pts


def recall_at_cost(pts, budget):
    """Interpolate recall at a common scan-cost budget (log-x linear interp)."""
    pts = sorted(pts)
    xs = np.log([max(c, 1.0) for c, _, _ in pts])
    ys = [r for _, r, _ in pts]
    xb = np.log(max(budget, 1.0))
    if xb <= xs[0] or xb >= xs[-1]:
        return float(np.interp(xb, xs, ys))
    return float(np.interp(xb, xs, ys))


def validate():
    """Does the fixed+origin arm reproduce dyf's own tree?"""
    from dyf.dyf_tree import build_dyf_tree

    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    sub = E[rng.choice(len(E), 40000, replace=False)]
    t0 = time.time()
    mine = build(sub, None, S.MAX_DEPTH, S.MIN_LEAF, "fixed", "origin", S.NUM_BITS, rng)
    t1 = time.time()
    theirs_tree = build_dyf_tree(sub, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED)
    t2 = time.time()

    def count_leaves(t):
        return 1 if not t["children"] else sum(count_leaves(c) for c in t["children"])

    print(f"numpy harness : {n_leaves(mine):>6} leaves  [{t1 - t0:.1f}s]")
    print(f"dyf build_tree: {count_leaves(theirs_tree):>6} leaves  [{t2 - t1:.1f}s]")
    print(f"  balance: {balance_stats(mine)}")
    print(
        "\nIf leaf counts are within a few percent the harness is a fair stand-in;\n"
        "the comparison below is internal to the harness in any case."
    )


TARGET_LEAVES = 13300  # the fixed4/origin baseline's leaf count

# min_leaf values that land each configuration near TARGET_LEAVES, found by the binary
# search below (`--search`). Cached because a derived-bits build costs ~60s and the search
# needs 5-6 of them. Re-derive with `--search` if the corpus or tree params change.
MATCHED_MIN_LEAF = {
    "fixed4_origin": 15,  # -> 13657 leaves
    "fixed4_median": 45,  # -> 12990 leaves
    "derived_origin": 10,  # -> 13408 leaves
    "derived_median": 12,  # -> verified at run time
}


def build_to_target(E, bm, om, target=TARGET_LEAVES, tol=0.06, max_tries=8):
    """Binary-search min_leaf so this configuration lands near `target` leaves.

    REQUIRED CONTROL. At identical params the median-offset arm produces 43,414 leaves vs
    13,327 for origin -- balanced cuts survive the leaf floor far more often. Comparing
    those directly measures leaf GRANULARITY, not the offset: finer leaves reach the same
    recall with ~9x fewer scanned candidates while costing proportionally more centroid
    comparisons. Holding leaf count fixed isolates the one variable under test.
    """
    lo, hi = 4, 4096
    best: tuple = ()
    for _ in range(max_tries):
        ml = int(round((lo * hi) ** 0.5))
        flat = build(E, None, S.MAX_DEPTH, ml, bm, om, S.NUM_BITS, np.random.default_rng(SEED))
        nl = n_leaves(flat)
        if not best or abs(nl - target) < abs(best[1] - target):
            best = (flat, nl, ml)
        print(f"    min_leaf={ml:<5} -> {nl:>7} leaves", flush=True)
        if abs(nl - target) / target <= tol:
            return flat, nl, ml
        if nl > target:
            lo = ml + 1
        else:
            hi = max(ml - 1, lo)
        if lo >= hi:
            break
    return best


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(E), NQ, replace=False)
    Qe = E[qi]
    print(f"corpus {E.shape}, {NQ} queries, k={K}, target leaves={TARGET_LEAVES}", flush=True)
    t0 = time.time()
    truth = S.exact_knn(Qe, E, k=K)
    print(f"exact kNN [{time.time() - t0:.0f}s]", flush=True)

    conds = [
        ("fixed4_origin", "fixed", "origin"),
        ("fixed4_median", "fixed", "median"),
        ("derived_origin", "derived", "origin"),
        ("derived_median", "derived", "median"),
    ]
    results = {}
    do_search = "--search" in sys.argv
    for name, bm, om in conds:
        t0 = time.time()
        if do_search or name not in MATCHED_MIN_LEAF:
            print(f"  searching min_leaf for {name}:", flush=True)
            flat, nl, ml = build_to_target(E, bm, om)
        else:
            ml = MATCHED_MIN_LEAF[name]
            flat = build(E, None, S.MAX_DEPTH, ml, bm, om, S.NUM_BITS, np.random.default_rng(SEED))
            nl = n_leaves(flat)
            print(f"  {name}: cached min_leaf={ml} -> {nl} leaves", flush=True)
        bt = time.time() - t0
        st = balance_stats(flat)
        pts = evaluate(E, flat, Qe, truth)
        results[name] = {
            "leaves": nl,
            "min_leaf": ml,
            "build_s": bt,
            **st,
            "curve": [{"cost": c, "recall": r, "probe": p} for c, r, p in pts],
        }
        print(
            f"{name:<16} leaves={nl:>6} (min_leaf={ml}) mean_bits={st['mean_bits']:.2f} "
            f"eff_buckets={st['mean_eff_buckets']:.2f} search={bt:.0f}s "
            f"bits_hist={st['bits_hist']}",
            flush=True,
        )

    print("\n" + "=" * 78)
    print("Leaf counts after matching (comparison is only valid if these are close):")
    for name, _, _ in conds:
        print(f"  {name:<16}{results[name]['leaves']:>8} leaves  min_leaf={results[name]['min_leaf']}")
    spread = max(results[n]["leaves"] for n, _, _ in conds) / min(results[n]["leaves"] for n, _, _ in conds)
    print(f"  max/min leaf ratio = {spread:.2f}  ({'OK' if spread < 1.25 else 'TOO WIDE -- read with care'})")

    print("\n" + "=" * 78)
    print("Recall@10 at MATCHED SCAN COST (mean candidates examined; leaf counts matched)")
    print("=" * 78)
    lo = max(min(c["cost"] for c in r["curve"]) for r in results.values())
    hi = min(max(c["cost"] for c in r["curve"]) for r in results.values())
    budgets = np.unique(np.round(np.geomspace(lo, hi, 6)).astype(int))
    print(f"{'condition':<16}" + "".join(f"{b:>11}" for b in budgets))
    for name, _, _ in conds:
        line = f"{name:<16}"
        for b in budgets:
            line += (
                f"{recall_at_cost([(c['cost'], c['recall'], c['probe']) for c in results[name]['curve']], b):>11.4f}"
            )
        print(line)
    base = "fixed4_origin"
    print(f"\ndelta vs {base}:")
    for name, _, _ in conds:
        if name == base:
            continue
        d = [
            recall_at_cost([(c["cost"], c["recall"], c["probe"]) for c in results[name]["curve"]], b)
            - recall_at_cost([(c["cost"], c["recall"], c["probe"]) for c in results[base]["curve"]], b)
            for b in budgets
        ]
        print(f"  {name:<16}" + "".join(f"{x:>+11.4f}" for x in d))

    path = os.path.join(S.CACHE, "derived_bits_results.json")
    with open(path, "w") as f:
        json.dump({"budgets": budgets.tolist(), "results": results}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
    else:
        main()
