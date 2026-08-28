"""Do spectral shapes follow patterns DOWN the tree -- by depth, and along root-leaf paths?

sec_cell_spectra.py measured shape per cell and found it regionally organised (46.6% of
depth-2 variance explained by the depth-1 parent). It never looked at how shape evolves
with DEPTH, or whether paths follow recurring trajectories.

THE CONFOUND, AND THE NULL THAT KILLS IT. Node size shrinks with depth, and effective rank
tracks sample size (12.4 at n=16 -> 111 at n=16384; `sec_cell_spectra.py --leafdemo`). So
any raw depth profile is mostly a size profile. Two controls used together:

  1. n-MATCHED descriptors: every node subsampled to N_SUB, so depths are comparable.
  2. PARENT-SUBSAMPLE NULL: compare each child's spectrum to a random subsample OF ITS OWN
     PARENT at exactly the same size. This is the decisive one -- it asks whether the SPLIT
     changed the shape, versus what drawing that many points from the parent would do
     anyway. delta ~ 0 means the partition is self-similar across scales; delta < 0 on
     eff_rank means the split genuinely concentrates structure.

Why it matters beyond description: `sec_multicorpus_bits.py` killed derived `num_bits`
because parallel analysis asks "is this component statistically real?" when an index needs
"is this split useful?". The parent-subsample delta measures SPLIT QUALITY directly rather
than per-node dimensionality, so if it decays toward zero with depth it says deeper splits
stop concentrating anything -- a stopping criterion of the kind PA failed to provide.

Run on two corpora (SEC 768d text, CMU MoCap 62d motion) because the single-corpus lesson
from sec_multicorpus_bits.py was expensive.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_spectra import descriptors, spectrum  # noqa: E402
from sec_cell_volume import flatten_ev, spearman  # noqa: E402

N_SUB = 300  # common sample size; nodes below this are skipped
N_DRAWS = 5
MAX_DEPTH_LOOK = 4
DESCS = ["eff_rank", "top1", "alpha"]
SEED = 42
PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
MAX_N = 120000


def node_depths(flat):
    """depth-from-root for every node id."""
    d = {0: 0}
    stack = [0]
    while stack:
        nid = stack.pop()
        for k in flat[nid]["children"]:
            d[k] = d[nid] + 1
            stack.append(k)
    return d


def parent_map(flat):
    p = {}
    for nid, nd in enumerate(flat):
        for k in nd["children"]:
            p[k] = nid
    return p


def analyse(E, name, out):
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    t0 = time.time()
    flat = flatten_ev(
        build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), E
    )
    dep = node_depths(flat)
    par = parent_map(flat)
    print(
        f"\n=== {name}: {E.shape}, {len(flat)} nodes, {S.n_leaves(flat)} leaves [{time.time() - t0:.0f}s] ===",
        flush=True,
    )

    # ---- per-node n-matched descriptors -------------------------------------------
    desc = {}
    for nid, nd in enumerate(flat):
        if dep[nid] > MAX_DEPTH_LOOK or len(nd["indices"]) < N_SUB:
            continue
        p = spectrum(E[nd["indices"]], N_SUB, rng, N_DRAWS)
        if p is not None:
            desc[nid] = descriptors(p)

    print(f"  {len(desc)} nodes with n >= {N_SUB}")
    print(f"\n  (A) depth profile at MATCHED n={N_SUB}  [raw shape by depth]")
    print(f"  {'depth':>6}{'nodes':>7}{'median n':>10}" + "".join(f"{d:>12}" for d in DESCS))
    prof = {}
    for d in range(MAX_DEPTH_LOOK + 1):
        ids = [i for i in desc if dep[i] == d]
        if not ids:
            continue
        row = {dn: float(np.mean([desc[i][dn] for i in ids])) for dn in DESCS}
        row["nodes"] = len(ids)
        row["median_n"] = float(np.median([len(flat[i]["indices"]) for i in ids]))
        prof[d] = row
        print(f"  {d:>6}{len(ids):>7}{row['median_n']:>10.0f}" + "".join(f"{row[dn]:>12.3f}" for dn in DESCS))

    # ---- parent-subsample null: did the SPLIT change the shape? --------------------
    print("\n  (B) child vs RANDOM SAME-SIZE SUBSAMPLE OF ITS PARENT  [split effect]")
    print(f"  {'depth':>6}{'pairs':>7}" + "".join(f"{'d_' + d:>14}" for d in DESCS) + f"{'sig?':>8}")
    deltas = {}
    for d in range(1, MAX_DEPTH_LOOK + 1):
        kids = [i for i in desc if dep[i] == d and par.get(i) is not None]
        rows = []
        for kid in kids:
            pid = par[kid]
            pidx = flat[pid]["indices"]
            if len(pidx) < N_SUB:
                continue
            # null: same-size draw from the parent, spectrum computed identically
            pool = pidx[rng.choice(len(pidx), min(len(pidx), max(N_SUB * 3, 2000)), replace=False)]
            pn = spectrum(E[pool], N_SUB, rng, N_DRAWS)
            if pn is None:
                continue
            nulld = descriptors(pn)
            rows.append({dn: desc[kid][dn] - nulld[dn] for dn in DESCS})
        if not rows:
            continue
        agg = {}
        for dn in DESCS:
            v = np.array([r[dn] for r in rows])
            agg[dn] = (float(v.mean()), float(v.std() / max(np.sqrt(len(v)), 1)))
        deltas[d] = {"n_pairs": len(rows), **{k: agg[k] for k in DESCS}}
        sig = abs(agg["eff_rank"][0]) > 2 * agg["eff_rank"][1]
        print(
            f"  {d:>6}{len(rows):>7}"
            + "".join(f"{agg[dn][0]:>+9.2f}+-{agg[dn][1]:<4.2f}" for dn in DESCS)
            + f"{'YES' if sig else 'no':>8}"
        )

    # ---- persistence along a path --------------------------------------------------
    print("\n  (C) does a node's shape predict its child's? (parent->child, n-matched)")
    for d in range(1, MAX_DEPTH_LOOK + 1):
        pairs = [(par[i], i) for i in desc if dep[i] == d and par.get(i) in desc]
        if len(pairs) < 8:
            continue
        line = f"  depth {d - 1}->{d}  pairs={len(pairs):>5}  "
        for dn in DESCS:
            r = spearman([desc[a][dn] for a, _ in pairs], [desc[b][dn] for _, b in pairs])
            line += f"rho({dn})={r:+.3f}  "
        print(line)

    out[name] = {
        "profile": prof,
        "split_delta": deltas,
        "n_nodes": len(desc),
        "leaves": S.n_leaves(flat),
    }


def main():
    out = {}
    E, *_ = S.load()
    analyse(E, "sec_768", out)

    p = os.path.join(PAPER, "cmu_mocap_features.npy")
    if os.path.exists(p):
        rng = np.random.default_rng(SEED)
        X = np.load(p).astype(np.float32)
        if len(X) > MAX_N:
            X = X[rng.choice(len(X), MAX_N, replace=False)]
        X = np.ascontiguousarray(X)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        analyse(X, "cmu_mocap_62", out)
    else:
        print(f"SKIP cmu_mocap: {p} not found")

    path = os.path.join(S.CACHE, "depth_spectra_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
