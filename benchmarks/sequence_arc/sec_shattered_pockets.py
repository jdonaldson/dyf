"""Are the non-self-similar 'pocket' splits fragments of ONE concept, shattered by the tree?

sec_nonselfsimilar.py found that the extreme concentrating splits pull small (1-10% share)
pockets of near-duplicate boilerplate out of much larger parents -- e.g. forward-looking
statement disclaimers (dup_frac 0.52-0.57) carved out of a risk_factors parent (0.01).

A root-to-leaf path is an intersection of halfspaces, so the tree CAN represent a convex
region. What it cannot represent is a concept that is multi-modal or non-convex in
embedding space: that must be shattered across branches. If the same boilerplate is carved
out independently under several different parents, those fragments are merge candidates.

THE RIGHT NULL IS CONDITIONAL ON TREE DISTANCE. Content similarity naturally decays with
tree distance -- that is what the tree is for -- so "these two cells are similar" is not
evidence of anything. A shattered fragment is a pair whose similarity is high *given how
far apart it sits*, i.e. an outlier against the LCA-depth-conditional similarity
distribution. LCA depth 0 = the pair diverges at the root, maximally distant.

DECISIVE CONTENT CHECK. Centroid cosine can be high for merely-similar cells. The test that
distinguishes "similar" from "the same thing twice" is the CROSS-PAIR DUPLICATE RATE: what
fraction of points in A have a near-duplicate (cos > 0.99) in B. Fragments of one concept
score high; genuinely distinct-but-similar topics do not.

IS IT ALREADY SOLVED? dyf merges leaves with Louvain over leaf centroids
(`agglomerate.louvain_cluster_leaves`, kNN graph, similarity_threshold=0.5). Every
candidate is checked against that threshold, so the output separates "fragments the
existing merger would already reunite" from "fragments it would miss".
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_volume import flatten_ev  # noqa: E402
from sec_depth_spectra import node_depths  # noqa: E402

SEED = 42
POCKET_Z = -30.0  # concentrating tail
POCKET_SHARE = 0.15
LOUVAIN_THRESHOLD = 0.5  # agglomerate.py default
DUP_COS = 0.99
SAMPLE_PER_CELL = 400
TOP_PAIRS = 10


def ancestors(flat, dep):
    par = {}
    for nid, nd in enumerate(flat):
        for k in nd["children"]:
            par[k] = nid

    def path(nid):
        p = [nid]
        while nid in par:
            nid = par[nid]
            p.append(nid)
        return p[::-1]  # root first

    return {nid: path(nid) for nid in range(len(flat))}, par


def lca_depth(pa, pb):
    d = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        d += 1
    return d - 1  # depth of the last shared node


def cross_dup(E, ia, ib, rng):
    """Fraction of A's sampled points having a near-duplicate in B."""
    a = E[rng.choice(ia, min(len(ia), SAMPLE_PER_CELL), replace=False)]
    b = E[rng.choice(ib, min(len(ib), SAMPLE_PER_CELL), replace=False)]
    return float(((a @ b.T).max(1) > DUP_COS).mean())


def main():
    E, D, T, SEC, Q = S.load()
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    flat = flatten_ev(
        build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), E
    )
    dep = node_depths(flat)
    paths, _ = ancestors(flat, dep)

    nss = json.load(open(os.path.join(S.CACHE, "nonselfsimilar_results.json")))["rows"]
    by_id = {r["child"]: r for r in nss}
    ids = sorted(by_id)
    print(f"tree {len(flat)} nodes; {len(ids)} scored cells", flush=True)

    cent = {}
    for nid in ids:
        idx = flat[nid]["indices"]
        c = E[idx].mean(0)
        cent[nid] = c / (np.linalg.norm(c) + 1e-12)

    pockets = [i for i in ids if by_id[i]["z_eff_rank"] < POCKET_Z and by_id[i]["share"] < POCKET_SHARE]
    others = [i for i in ids if i not in set(pockets)]
    print(f"pockets (z_eff<{POCKET_Z}, share<{POCKET_SHARE}): {len(pockets)}   other cells: {len(others)}")

    # ---- similarity conditional on tree distance -----------------------------------
    rows = []
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "cos": float(cent[a] @ cent[b]),
                    "lca": lca_depth(paths[a], paths[b]),
                    "both_pocket": (a in set(pockets)) and (b in set(pockets)),
                }
            )
    print(f"\n{len(rows)} cell pairs")
    print("\nSimilarity conditional on tree distance (the null a merge candidate must beat):")
    print(f"{'LCA depth':>10}{'pairs':>8}{'mean cos':>10}{'p95 cos':>9}{'max cos':>9}{'pocket-pair mean':>18}")
    stats = {}
    for L in sorted({r["lca"] for r in rows}):
        sub = [r["cos"] for r in rows if r["lca"] == L]
        pk = [r["cos"] for r in rows if r["lca"] == L and r["both_pocket"]]
        stats[L] = (float(np.mean(sub)), float(np.percentile(sub, 95)))
        print(
            f"{L:>10}{len(sub):>8}{np.mean(sub):>10.3f}{np.percentile(sub, 95):>9.3f}"
            f"{max(sub):>9.3f}{(np.mean(pk) if pk else float('nan')):>18.3f}"
        )

    # ---- candidates: similar DESPITE distance --------------------------------------
    for r in rows:
        mu, p95 = stats[r["lca"]]
        r["excess"] = r["cos"] - p95

    cand = sorted([r for r in rows if r["lca"] <= 1 and r["excess"] > 0], key=lambda r: -r["cos"])[:TOP_PAIRS]
    print("\n=== top merge candidates: high similarity DESPITE diverging at depth <=1 ===")
    print(f"{'A':>7}{'B':>7}{'lca':>5}{'cos':>7}{'xdup A>B':>10}{'xdup B>A':>10}{'louvain?':>10}  content")
    audit = []
    for r in cand:
        ia, ib = flat[r["a"]]["indices"], flat[r["b"]]["indices"]
        xa, xb = cross_dup(E, ia, ib, rng), cross_dup(E, ib, ia, rng)
        sa, sb = SEC[ia], SEC[ib]
        va, ca_ = np.unique(sa, return_counts=True)
        vb, cb_ = np.unique(sb, return_counts=True)
        linked = r["cos"] >= LOUVAIN_THRESHOLD
        audit.append({**r, "xdup_ab": xa, "xdup_ba": xb, "louvain_would_link": linked})
        print(
            f"{r['a']:>7}{r['b']:>7}{r['lca']:>5}{r['cos']:>7.3f}{xa:>10.2f}{xb:>10.2f}"
            f"{('YES' if linked else 'no'):>10}  "
            f"A={va[ca_.argmax()]}({ca_.max() / ca_.sum():.2f},n={len(ia)}) "
            f"B={vb[cb_.argmax()]}({cb_.max() / cb_.sum():.2f},n={len(ib)})"
        )

    # ---- are pockets over-represented among distant-but-similar pairs? -------------
    far = [r for r in rows if r["lca"] <= 1]
    hi = [r for r in far if r["excess"] > 0]
    pf = np.mean([r["both_pocket"] for r in far]) if far else 0
    ph = np.mean([r["both_pocket"] for r in hi]) if hi else 0
    print(
        f"\nAmong pairs diverging at depth <=1: {len(far)} pairs, {100 * pf:.1f}% are pocket-pairs.\n"
        f"Among those ALSO above the depth-conditional p95: {len(hi)} pairs, {100 * ph:.1f}% are pocket-pairs."
    )
    print(f"  enrichment = {(ph / pf if pf > 0 else float('nan')):.2f}x")

    # ---- would Louvain already reunite them? ---------------------------------------
    linked = sum(1 for a in audit if a["louvain_would_link"])
    print(
        f"\nOf the {len(audit)} top candidates, {linked} exceed the Louvain centroid threshold "
        f"({LOUVAIN_THRESHOLD}) and would already be merged by agglomerate.louvain_cluster_leaves.\n"
        f"The remaining {len(audit) - linked} would be MISSED by the existing merger."
    )

    path = os.path.join(S.CACHE, "shattered_pockets_results.json")
    with open(path, "w") as f:
        json.dump({"candidates": audit, "lca_stats": {str(k): v for k, v in stats.items()}}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
