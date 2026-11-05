"""Does the norm inside each leaf (the coordinate the tree normalises away) carry
structure on the seven-dataset bench?  Follows the 2-D shapes probe
(scratchpad wedge_radius_probe.py, 2026-11-04) where re-reading the norm inside
each angular wedge recovered concentric rings exactly.

Per dataset, shipped tree with library defaults (3, 4, 20):
  1. per leaf: split sorted norms at the best Otsu cut, accept only if the bands are
     >= SEP_MIN pooled std apart (recursive, depth <= 3). -> (leaf, band) sub-nodes.
  2. purity vs labels: plain leaves / bands / size-matched random split null.
  3. Louvain (dyf_rs) on four graphs, no point reassignment, res 1.0, k=10:
       cosine kNN on plain leaves (the shipped stage), cosine on bands,
       Euclidean Gaussian-weighted kNN on plain leaves, Euclidean on bands.
  4. ring proxy: inner/outer mean-norm ratio across split leaves.  A global
     concentric structure gives one tight ratio (circles: 0.5 everywhere); a
     per-leaf nuisance (library size) gives a broad spread.
Outcome variables fixed before running: purity gain over the null, ARI/AMI.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python wedge_radius_bench.py [names...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors

from dyf import build_dyf_tree

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
SEP_MIN, K = 3.0, 10
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def load(name):
    meta = json.loads((BENCH / "datasets.json").read_text())[name]
    X = np.ascontiguousarray(np.load(meta.get("x_path", BENCH / f"{name}_X.npy")), dtype=np.float32)
    y = np.load(meta.get("y_path", BENCH / f"{name}_y.npy"), allow_pickle=True)
    _, y = np.unique(y, return_inverse=True)
    return X, y, meta


def leaf_map(tree, n):
    item_leaf = np.full(n, -1, dtype=np.int64)
    nleaf = 0

    def walk(node):
        nonlocal nleaf
        ch = node.get("children") or []
        if not ch:
            item_leaf[np.asarray(node["indices"])] = nleaf
            nleaf += 1
            return
        for c in ch:
            walk(c)

    walk(tree)
    assert (item_leaf >= 0).all()
    return item_leaf


def purity(groups, y):
    tot = 0
    for g in np.unique(groups):
        tot += np.bincount(y[groups == g]).max()
    return tot / len(y)


def split_1d(r, sep_min, min_band, depth=0, max_depth=3):
    n = len(r)
    out = np.zeros(n, dtype=np.int64)
    if n < 2 * min_band or depth >= max_depth:
        return out
    order = np.argsort(r)
    rs = r[order].astype(np.float64)
    cs, cs2 = np.cumsum(rs), np.cumsum(rs * rs)
    i = np.arange(min_band, n - min_band + 1)
    n1, n2 = i, n - i
    m1 = cs[i - 1] / n1
    m2 = (cs[-1] - cs[i - 1]) / n2
    v1 = np.maximum(cs2[i - 1] / n1 - m1 * m1, 1e-12)
    v2 = np.maximum((cs2[-1] - cs2[i - 1]) / n2 - m2 * m2, 1e-12)
    sep = (m2 - m1) / np.sqrt((v1 + v2) / 2)
    j = int(np.argmax(sep))
    if sep[j] < sep_min:
        return out
    cut = i[j]
    lo, hi = order[:cut], order[cut:]
    out[lo] = split_1d(r[lo], sep_min, min_band, depth + 1, max_depth)
    out[hi] = split_1d(r[hi], sep_min, min_band, depth + 1, max_depth) + out[lo].max() + 1
    return out


def refine(X, item_leaf):
    r = np.linalg.norm(X, axis=1)
    sub = np.zeros(len(X), dtype=np.int64)
    nxt, split_leaves, ratios = 0, [], []
    for li in np.unique(item_leaf):
        m = np.flatnonzero(item_leaf == li)
        bands = split_1d(r[m], SEP_MIN, max(5, len(m) // 20))
        if bands.max() > 0:
            split_leaves.append(li)
            means = sorted(r[m][bands == b].mean() for b in np.unique(bands))
            ratios.append(means[0] / means[-1])
        sub[m] = bands + nxt
        nxt += bands.max() + 1
    return sub, np.asarray(split_leaves), np.asarray(ratios)


def null_purity(sub, item_leaf, y, draws=3, seed=0):
    """Same band sizes inside each leaf, random membership."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        s = sub.copy()
        for li in np.unique(item_leaf):
            m = np.flatnonzero(item_leaf == li)
            s[m] = rng.permutation(sub[m])
        vals.append(purity(s, y))
    return float(np.mean(vals))


def centroids(X, groups):
    ids = np.unique(groups)
    C = np.stack([X[groups == g].mean(0) for g in ids])
    remap = np.searchsorted(ids, groups)
    return C, remap


def louvain_cosine(C):
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    adj = build_knn_graph(np.ascontiguousarray(Cn, dtype=np.float32), k=K)
    edges = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
    lab, nc = louvain_communities(len(C), edges, resolution=1.0)
    return np.asarray(lab), nc


def louvain_euclid(C):
    kk = min(K + 1, len(C))
    D, J = NearestNeighbors(n_neighbors=kk).fit(C).kneighbors(C)
    sigma = np.median(D[:, 1:]) + 1e-12
    W = np.exp(-(D[:, 1:] ** 2) / sigma**2)
    edges = [(i, int(J[i, p]), float(W[i, p])) for i in range(len(C)) for p in range(kk - 1)]
    lab, nc = louvain_communities(len(C), edges, resolution=1.0)
    return np.asarray(lab), nc


def run(name):
    t0 = time.perf_counter()
    X, y, meta = load(name)
    tree = build_dyf_tree(X, seed=42, **TREE)
    item_leaf = leaf_map(tree, len(X))
    n_leaves = len(np.unique(item_leaf))
    sub, split_leaves, ratios = refine(X, item_leaf)
    res = dict(
        name=name,
        n=len(X),
        d=X.shape[1],
        k_true=int(len(np.unique(y))),
        n_leaves=n_leaves,
        n_sub=int(len(np.unique(sub))),
        leaves_split=int(len(split_leaves)),
        points_in_split_leaves=float(np.isin(item_leaf, split_leaves).mean()),
        purity_leaves=purity(item_leaf, y),
        purity_bands=purity(sub, y),
        purity_null=null_purity(sub, item_leaf, y),
        ratio_median=float(np.median(ratios)) if len(ratios) else None,
        ratio_iqr=[float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))] if len(ratios) else None,
        norm_cv=float(np.std(np.linalg.norm(X, axis=1)) / np.mean(np.linalg.norm(X, axis=1))),
    )
    print(
        f"\n== {name}: n={res['n']} d={res['d']} k_true={res['k_true']} leaves={n_leaves} norm CV={res['norm_cv']:.3f}",
        flush=True,
    )
    print(
        f"   split {res['leaves_split']}/{n_leaves} leaves ({res['points_in_split_leaves']:.0%} of points) "
        f"-> {res['n_sub']} sub-nodes; purity leaves {res['purity_leaves']:.3f} bands {res['purity_bands']:.3f} "
        f"null {res['purity_null']:.3f}; inner/outer ratio median {res['ratio_median']} iqr {res['ratio_iqr']}",
        flush=True,
    )
    for label, groups in (("leaves", item_leaf), ("bands", sub)):
        C, remap = centroids(X, groups)
        for how, fn in (("cosine", louvain_cosine), ("euclid", louvain_euclid)):
            lab, nc = fn(C)
            pl = lab[remap]
            res[f"{label}_{how}"] = dict(k=int(nc), ari=float(ari(y, pl)), ami=float(ami(y, pl)))
            print(
                f"   {label:6s} {how}: k={nc:4d} ARI={res[f'{label}_{how}']['ari']:.3f} "
                f"AMI={res[f'{label}_{how}']['ami']:.3f}",
                flush=True,
            )
    res["seconds"] = time.perf_counter() - t0
    print(f"   ({res['seconds']:.1f}s)", flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "wedge_radius_bench_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
