"""Does dyf's final partition cut dense structure more than the incumbent?

A "sliced" region shows up as a community whose points' nearest neighbours mostly
live in other communities.  Measure that directly on a point-level kNN graph
(pynndescent k=15 Euclidean on X, the same graph the incumbent Leiden ran on).

Partitions compared per dataset (all from /Volumes/Models/dyf_bench_2026-09-27):
  dyf          shipped path, library defaults (sC_labels_<name>_default_3_4_20.npy)
  leiden@res   pynndescent+Leiden at the resolution whose k is closest to dyf's k
  kmeans@k     k-means told dyf's k (matched-k reference, one init)
  truth        the ground-truth labels (how much a correct partition must cut)
  leaves       dyf tree leaves before Louvain (how much the TREE slices)
  kmeans@L     MiniBatchKMeans with as many cells as the tree has leaves
Metrics:
  cut   fraction of directed kNN edges whose endpoints get different labels
  phi   per-community conductance = out-edges / min(vol S, vol of the rest), vol = k*|S|;
        reported as median, p90, max (with the size of the worst community)
Outcome variable fixed before running: dyf cut vs leiden/kmeans cut at matched k,
and which datasets (if any) show dyf communities with conductance well above both.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python community_conductance_bench.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from pynndescent import NNDescent
from sklearn.cluster import KMeans, MiniBatchKMeans

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


def knn_graph(name, X):
    cache = OUT / f"knn{KNN}_{name}.npy"
    if cache.exists():
        return np.load(cache)
    t0 = time.perf_counter()
    idx, _ = NNDescent(X, n_neighbors=KNN + 1, metric="euclidean", random_state=42, n_jobs=-1).neighbor_graph
    J = idx[:, 1:].astype(np.int32)
    np.save(cache, J)
    print(f"   kNN {KNN} via pynndescent: {time.perf_counter() - t0:.1f}s", flush=True)
    return J


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
    return item_leaf


def conductance(lab, J):
    """cut fraction + per-community conductance on the directed kNN graph."""
    _, lab = np.unique(lab, return_inverse=True)
    n, k = J.shape
    src = np.repeat(lab, k)
    dst = lab[J.ravel()]
    cross = src != dst
    cut = float(cross.mean())
    out_edges = np.bincount(src[cross], minlength=lab.max() + 1)
    size = np.bincount(lab, minlength=lab.max() + 1)
    vol = k * size
    phi = out_edges / np.maximum(np.minimum(vol, k * n - vol), 1)
    worst = int(np.argmax(phi))
    return dict(
        k=int(len(size)),
        cut=cut,
        phi_median=float(np.median(phi)),
        phi_p90=float(np.percentile(phi, 90)),
        phi_max=float(phi[worst]),
        worst_size=int(size[worst]),
        worst_frac=float(size[worst] / n),
    )


def fmt(tag, r):
    return (
        f"   {tag:14s} k={r['k']:5d} cut={r['cut']:.3f} phi med={r['phi_median']:.3f} "
        f"p90={r['phi_p90']:.3f} max={r['phi_max']:.3f} (worst {r['worst_frac']:.1%} of pts)"
    )


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    print(f"\n== {name}: n={len(X)} d={X.shape[1]}", flush=True)
    J = knn_graph(name, X)
    res = {}
    dyf_lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
    res["dyf"] = conductance(dyf_lab, J)
    print(fmt("dyf", res["dyf"]), flush=True)
    kd = res["dyf"]["k"]
    best = None
    for r in ("0.5", "1.0", "2.0", "4.0"):
        lab = np.load(BENCH / f"sB_labels_{name}_pynn_leiden_res{r}.npy")
        c = conductance(lab, J)
        res[f"leiden@{r}"] = c
        if best is None or abs(c["k"] - kd) < abs(res[best]["k"] - kd):
            best = f"leiden@{r}"
    print(fmt(best, res[best]), flush=True)
    km = KMeans(n_clusters=kd, n_init=1, random_state=42).fit_predict(X)
    res["kmeans@k"] = conductance(km, J)
    print(fmt("kmeans@k", res["kmeans@k"]), flush=True)
    res["truth"] = conductance(y, J)
    print(fmt("truth", res["truth"]), flush=True)
    leaves = leaf_map(build_dyf_tree(X, seed=42, **TREE), len(X))
    res["leaves"] = conductance(leaves, J)
    print(fmt("leaves", res["leaves"]), flush=True)
    L = res["leaves"]["k"]
    mbk = MiniBatchKMeans(n_clusters=L, batch_size=8192, max_iter=15, n_init=1, random_state=42).fit_predict(X)
    res["kmeans@L"] = conductance(mbk, J)
    print(fmt("kmeans@L", res["kmeans@L"]), flush=True)
    res["best_leiden"] = best
    res["seconds"] = time.perf_counter() - t0
    print(f"   ({res['seconds']:.1f}s)", flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "community_conductance_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
