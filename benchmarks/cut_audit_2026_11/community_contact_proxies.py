"""Can anything dyf already computes reproduce the point-kNN contact between final
communities (community_conductance_bench.py), without the point kNN graph?

Target per ordered community pair (c, d), from the cached pynndescent k=15 graph:
  share_cd   = kNN out-edges of c that land in d / all out-edges of c
  pair_cut   = (flow[c,d] + flow[d,c]) / (k * (|c| + |d|))     (symmetric contact)
  mutual     = min(share_cd, share_dc)
  slice      = both communities share the same majority ground-truth label

Free proxies (no point kNN graph):
  A  centroid tie : fraction of c's points whose nearest OTHER community centroid is d and
                    whose margin (own sim - d sim) is below the dataset's 20th percentile
                    of margins.  Uses exactly what _compute_point_metrics computes.
  B  leaf graph   : cosine k=10 leaf-centroid graph (the graph Louvain ran on), each leaf
                    assigned to its majority community; cross-edge weight c->d / c's total.
  C  tree margin  : as A, but the tie is counted only when the point's tree margin
                    (min over depths of its similarity to the child centroid it was routed
                    to, from point_margin_map) is in the bottom quintile.
Scores per dataset: Spearman rho between each proxy and share_cd over pairs with any
contact; overlap of the proxy's top-5 pairs with the kNN top-5 by share_cd; and AUC of
each feature for slice vs non-slice among the kNN top-10 contact pairs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from dyf import build_dyf_tree

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
KNN = 15
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def load(name):
    meta = json.loads((BENCH / "datasets.json").read_text())[name]
    X = np.ascontiguousarray(np.load(meta.get("x_path", BENCH / f"{name}_X.npy")), dtype=np.float32)
    y = np.load(meta.get("y_path", BENCH / f"{name}_y.npy"), allow_pickle=True)
    _, y = np.unique(y, return_inverse=True)
    return X, y


def walk_tree(tree, n):
    """leaf id per point, and per-point min margin (similarity to routed child centroid)."""
    item_leaf = np.full(n, -1, dtype=np.int64)
    margin = np.full(n, np.inf, dtype=np.float32)
    nleaf = 0

    def walk(node):
        nonlocal nleaf
        pm = node.get("point_margin_map")
        if pm:
            ids = np.fromiter(pm.keys(), dtype=np.int64, count=len(pm))
            vals = np.fromiter(pm.values(), dtype=np.float32, count=len(pm))
            margin[ids] = np.minimum(margin[ids], vals)
        ch = node.get("children") or []
        if not ch:
            item_leaf[np.asarray(node["indices"])] = nleaf
            nleaf += 1
            return
        for c in ch:
            walk(c)

    walk(tree)
    margin[~np.isfinite(margin)] = 1.0
    return item_leaf, margin


def knn_targets(lab, J, y):
    n, k = J.shape
    C = lab.max() + 1
    flow = np.zeros((C, C), dtype=np.int64)
    np.add.at(flow, (np.repeat(lab, k), lab[J.ravel()]), 1)
    size = np.bincount(lab, minlength=C)
    out = flow.sum(1) - np.diag(flow)
    share = flow / np.maximum(out, 1)[:, None]
    np.fill_diagonal(share, 0)
    pair_cut = (flow + flow.T) / (k * (size[:, None] + size[None, :]))
    np.fill_diagonal(pair_cut, 0)
    maj = np.array([np.bincount(y[lab == c]).argmax() for c in range(C)])
    same = maj[:, None] == maj[None, :]
    return share, pair_cut, same, size


def proxy_centroid_tie(X, lab, tree_margin=None):
    C = lab.max() + 1
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    cent = np.stack([Xn[lab == c].mean(0) for c in range(C)])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    sims = Xn @ cent.T
    own = sims[np.arange(len(X)), lab]
    sims[np.arange(len(X)), lab] = -np.inf
    other = sims.argmax(1)
    margin = own - sims[np.arange(len(X)), other]
    tie = margin < np.percentile(margin, 20)
    if tree_margin is not None:
        tie &= tree_margin < np.percentile(tree_margin, 20)
    M = np.zeros((C, C))
    np.add.at(M, (lab[tie], other[tie]), 1)
    M /= np.maximum(np.bincount(lab, minlength=C), 1)[:, None]
    return M, cent


def proxy_leaf_graph(X, lab, item_leaf):
    C = lab.max() + 1
    L = item_leaf.max() + 1
    cent = np.stack([X[item_leaf == l].mean(0) for l in range(L)])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    adj = build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10)
    leaf_comm = np.array([np.bincount(lab[item_leaf == l]).argmax() for l in range(L)])
    M = np.zeros((C, C))
    for i, nb in enumerate(adj):
        for j, w in nb:
            M[leaf_comm[i], leaf_comm[int(j)]] += w
    tot = M.sum(1)
    np.fill_diagonal(M, 0)
    return M / np.maximum(tot, 1e-12)[:, None]


def score(name, share, pair_cut, same, proxies):
    C = share.shape[0]
    off = ~np.eye(C, dtype=bool)
    contact = off & (share > 0)
    target = share[contact]
    top_knn = set(map(tuple, np.argwhere(off)[np.argsort(-share[off])[:5]]))
    # slice vs not among kNN top-10 contact pairs
    top10 = np.argwhere(off)[np.argsort(-share[off])[:10]]
    is_slice = np.array([same[c, d] for c, d in top10])
    res = {"n_pairs_with_contact": int(contact.sum()), "n_slices_in_top10": int(is_slice.sum())}
    print(
        f"\n== {name}: k={C}, {contact.sum()} pairs with kNN contact; "
        f"{is_slice.sum()}/10 of the top-10 contact pairs are slices (same label)"
    )
    feats = {"knn_mutual": np.minimum(share, share.T), "knn_pair_cut": pair_cut, **proxies}
    for fname, M in feats.items():
        rho = spearmanr(M[contact], target).correlation if fname not in ("knn_mutual", "knn_pair_cut") else None
        top_p = set(map(tuple, np.argwhere(off)[np.argsort(-M[off])[:5]]))
        overlap = len(top_p & top_knn)
        vals = np.array([M[c, d] for c, d in top10])
        auc = roc_auc_score(is_slice, vals) if 0 < is_slice.sum() < len(is_slice) else None
        res[fname] = dict(
            rho=None if rho is None else float(rho), top5_overlap=overlap, slice_auc=None if auc is None else float(auc)
        )
        print(
            f"   {fname:14s} rho vs kNN share={'  n/a' if rho is None else f'{rho:5.2f}'}  "
            f"top-5 overlap={overlap}/5  slice-vs-leak AUC in top-10={'n/a' if auc is None else f'{auc:.2f}'}"
        )
    return res


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn{KNN}_{name}.npy")
    lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
    _, lab = np.unique(lab, return_inverse=True)
    share, pair_cut, same, size = knn_targets(lab, J, y)
    item_leaf, tree_margin = walk_tree(build_dyf_tree(X, seed=42, **TREE), len(X))
    A, cent = proxy_centroid_tie(X, lab)
    Cm, _ = proxy_centroid_tie(X, lab, tree_margin)
    B = proxy_leaf_graph(X, lab, item_leaf)
    csim = cent @ cent.T
    np.fill_diagonal(csim, 0)
    res = score(
        name,
        share,
        pair_cut,
        same,
        {"A_centroid_tie": A, "B_leaf_graph": B, "C_tie_x_treemargin": Cm, "D_centroid_cos": csim},
    )
    res["seconds"] = time.perf_counter() - t0
    print(f"   ({res['seconds']:.1f}s)", flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "community_contact_proxies_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
