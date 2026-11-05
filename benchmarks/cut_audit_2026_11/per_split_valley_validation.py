"""Does the cut profile sort real boundaries from continuum slices, per split of the nested
profile?  Population test (every touching sibling pair at every level), not a top-k slice.

Per (dataset, seed): library-default tree, shipped cosine k=10 leaf graph, resolution sweep,
snapped bottom-up into exact nesting (nested_sweep_snap_test.py).  For each consecutive
level pair and each parent community that splits, every pair of children that touch on the
point kNN graph (cross edges >= 2% of the smaller child's out-edges) gets the cut profile
(community_cut_profile_ownmode.profile2: valley, c_side, d_side, own_offset).
Ground truth per sibling pair (labels used only here):
  tv          total-variation distance between the two children's label distributions
              (0 = same composition, 1 = disjoint) -- graded
  different   majority labels differ -- binary
Scores: Spearman(valley, tv) (expect negative: deeper valley -> more different), AUC of
-valley and of -min(c_side, d_side) for `different`, per dataset and pooled.
Pre-registered bar: pooled AUC >= 0.70 to call it usable as a profile annotation.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python per_split_valley_validation.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from dyf import build_dyf_tree

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402
from community_cut_profile_ownmode import profile2  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
TOUCH = 0.02
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def relabel(lab):
    return np.unique(lab, return_inverse=True)[1]


def snapped_profile(X, item_leaf, edges, L):
    flat = {}
    for res in RES:
        lab, _ = louvain_communities(L, edges, resolution=res)
        flat[res] = relabel(np.asarray(lab))[item_leaf]
    snapped = {RES[-1]: flat[RES[-1]]}
    for i in range(len(RES) - 2, -1, -1):
        finer = snapped[RES[i + 1]]
        parent = np.zeros(finer.max() + 1, dtype=np.int64)
        for f in np.unique(finer):
            parent[f] = np.bincount(flat[RES[i]][finer == f]).argmax()
        snapped[RES[i]] = relabel(parent[finer])
    return snapped


def label_dist(y, m, n_labels):
    return np.bincount(y[m], minlength=n_labels) / max(m.sum(), 1)


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    n, n_labels = len(X), int(y.max() + 1)
    J = np.load(OUT / f"knn15_{name}.npy")
    k = J.shape[1]
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), n)
    L = item_leaf.max() + 1
    Xn = unit(X)
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    adj = build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10)
    edges = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
    prof = snapped_profile(X, item_leaf, edges, L)
    rows = []
    for i in range(len(RES) - 1):
        coarse, fine = prof[RES[i]], prof[RES[i + 1]]
        # contact between fine communities on the point graph
        C = fine.max() + 1
        flow = np.zeros((C, C), dtype=np.int64)
        np.add.at(flow, (np.repeat(fine, k), fine[J.ravel()]), 1)
        out = flow.sum(1) - np.diag(flow)
        for p in np.unique(coarse):
            kids = np.unique(fine[coarse == p])
            if len(kids) < 2:
                continue
            for a_i in range(len(kids)):
                for b_i in range(a_i + 1, len(kids)):
                    a, b = int(kids[a_i]), int(kids[b_i])
                    share = (flow[a, b] + flow[b, a]) / max(min(out[a], out[b]), 1)
                    if share < TOUCH:
                        continue
                    pr = profile2(X, fine, a, b)
                    ma, mb = fine == a, fine == b
                    da, db = label_dist(y, ma, n_labels), label_dist(y, mb, n_labels)
                    rows.append(
                        dict(
                            level=i,
                            parent=int(p),
                            a=a,
                            b=b,
                            size_a=int(ma.sum()),
                            size_b=int(mb.sum()),
                            share=float(share),
                            valley=pr["valley"],
                            c_side=pr["c_side"],
                            d_side=pr["d_side"],
                            own_offset=pr["own_offset"],
                            tv=float(0.5 * np.abs(da - db).sum()),
                            different=bool(da.argmax() != db.argmax()),
                        )
                    )
    v = np.array([r["valley"] for r in rows])
    side = np.array([min(r["c_side"], r["d_side"]) for r in rows])
    tv = np.array([r["tv"] for r in rows])
    diff = np.array([r["different"] for r in rows])
    res = {
        "n_pairs": len(rows),
        "frac_different": float(diff.mean()) if len(rows) else None,
        "rho_valley_tv": float(spearmanr(v, tv).correlation) if len(rows) > 2 else None,
        "auc_valley": float(roc_auc_score(diff, -v)) if 0 < diff.sum() < len(diff) else None,
        "auc_side": float(roc_auc_score(diff, -side)) if 0 < diff.sum() < len(diff) else None,
        "pairs": rows,
    }
    print(
        f"\n== {name} seed={seed}: {len(rows)} touching sibling pairs over {len(RES) - 1} level steps; "
        f"different-majority {res['frac_different']:.0%}; Spearman(valley, tv)={res['rho_valley_tv']:.2f}; "
        f"AUC(different | -valley)={res['auc_valley']:.2f}, AUC(| -min side)={res['auc_side']:.2f}",
        flush=True,
    )
    # by level: valley mean where different vs same
    for lvl in range(len(RES) - 1):
        sel = [r for r in rows if r["level"] == lvl]
        if len(sel) < 5:
            continue
        vd = [r["valley"] for r in sel if r["different"]]
        vs = [r["valley"] for r in sel if not r["different"]]
        print(
            f"   level {RES[lvl]}->{RES[lvl + 1]}: {len(sel):4d} pairs; mean valley different={np.mean(vd) if vd else float('nan'):.2f} "
            f"same={np.mean(vs) if vs else float('nan'):.2f}",
            flush=True,
        )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "per_split_valley_validation_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
