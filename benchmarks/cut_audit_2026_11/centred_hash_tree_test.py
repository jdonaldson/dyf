"""Step 2: does centring the hash threshold (sign(w.(x - mu)) instead of sign(w.x)) change
anything?  The Rust kernel tests dot > 0 with no offset (dyf-core compute.rs), so leaves are
cones through the origin; PCA directions come from CENTRED data but the threshold is 0.

Python tree builder replicating the Rust recursion (normalise rows once; per node PCA top
num_bits directions; sign bits -> bucket id; stop at depth 0 or n < 2*min_leaf; children
below min_leaf become leaves), with `centred` switching the threshold.  Arms per (dataset,
seed): rust (shipped build_dyf_tree, control for the Python replica), py-uncentred,
py-centred; same downstream for all (cosine k=10 leaf-centroid Louvain, block and point).
Also the gallery shapes (moons / circles / blobs), where cones vs polytopes should matter most.
Outcome fixed before running: leaf purity and block/point ARI, centred vs uncentred, with
py-uncentred ~ rust as the check that the replica is faithful.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import conductance, leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def py_tree_leaves(Xn, centred, num_bits, max_depth, min_leaf_size, seed):
    """Leaf id per point from the Python replica of the recursion."""
    n = len(Xn)
    item_leaf = np.full(n, -1, dtype=np.int64)
    counter = [0]

    def make_leaf(idx):
        item_leaf[idx] = counter[0]
        counter[0] += 1

    def rec(idx, depth):
        if depth == 0 or len(idx) < min_leaf_size * 2:
            make_leaf(idx)
            return
        S = Xn[idx]
        k = min(num_bits, S.shape[1], len(idx) - 1)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(S)
        W = pca.components_.astype(np.float32)
        P = (S - pca.mean_) @ W.T if centred else S @ W.T
        bits = (P > 0).astype(np.int64)
        ids = (bits * (1 << np.arange(k))).sum(1)
        uniq = np.unique(ids)
        if len(uniq) <= 1:
            make_leaf(idx)
            return
        for b in uniq:
            child = idx[ids == b]
            if len(child) < min_leaf_size:
                make_leaf(child)
            else:
                rec(child, depth - 1)

    rec(np.arange(n), max_depth)
    return item_leaf


def purity(groups, y):
    return sum(np.bincount(y[groups == g]).max() for g in np.unique(groups)) / len(y)


def downstream(X, Xn, item_leaf, y, J):
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    leaf_lab, _, _ = _run_louvain_on_centroids(cent, k=10, resolution=1.0, similarity_threshold=0.5)
    _, leaf_lab = np.unique(np.asarray(leaf_lab), return_inverse=True)
    block = leaf_lab[item_leaf].astype(np.int32)
    point = _reassign_points(block.copy(), X)
    out = {"leaves": int(L), "leaf_purity": float(purity(item_leaf, y))}
    for tag, lab in (("block", block), ("point", point)):
        _, lab2 = np.unique(lab, return_inverse=True)
        c = (
            conductance(lab2, J)
            if J is not None
            else {"k": int(lab2.max() + 1), "cut": float("nan"), "phi_max": float("nan")}
        )
        c.update(ari=float(ari(y, lab2)), ami=float(ami(y, lab2)))
        out[tag] = c
    return out


def fmt(tag, r):
    return (
        f"   {tag:12s} leaves={r['leaves']:5d} purity={r['leaf_purity']:.3f} | block k={r['block']['k']:3d} "
        f"ARI={r['block']['ari']:.3f} cut={r['block']['cut']:.3f} | point k={r['point']['k']:3d} "
        f"ARI={r['point']['ari']:.3f} cut={r['point']['cut']:.3f}"
    )


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    Xn = unit(X)
    res = {}
    print(f"\n== {name} seed={seed}", flush=True)
    res["rust"] = downstream(X, Xn, leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X)), y, J)
    print(fmt("rust", res["rust"]), flush=True)
    for centred in (False, True):
        tag = "py-centred" if centred else "py-uncentred"
        res[tag] = downstream(X, Xn, py_tree_leaves(Xn, centred, seed=seed, **TREE), y, J)
        print(fmt(tag, res[tag]), flush=True)
    res["seconds"] = time.perf_counter() - t0
    return res


def shapes():
    out = {}
    sets = {
        "moons": make_moons(n_samples=3000, noise=0.08, random_state=42),
        "circles": make_circles(n_samples=3000, noise=0.05, factor=0.5, random_state=42),
        "blobs": make_blobs(
            n_samples=[200, 400, 200, 1000, 3000], cluster_std=[0.3, 0.8, 0.5, 1.5, 2.0], n_features=2, random_state=42
        ),
    }
    for name, (X, y) in sets.items():
        X = X.astype(np.float32)
        Xn = unit(X)
        print(f"\n== shapes/{name}", flush=True)
        out[name] = {}
        for centred in (False, True):
            tag = "py-centred" if centred else "py-uncentred"
            r = downstream(X, Xn, py_tree_leaves(Xn, centred, seed=42, **TREE), y, None)
            out[name][tag] = r
            print(
                f"   {tag:12s} leaves={r['leaves']:4d} purity={r['leaf_purity']:.3f} block ARI={r['block']['ari']:.3f} "
                f"point ARI={r['point']['ari']:.3f}",
                flush=True,
            )
    return out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "centred_hash_tree_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["shapes"] = shapes()
    out_path.write_text(json.dumps(results, indent=1))
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
