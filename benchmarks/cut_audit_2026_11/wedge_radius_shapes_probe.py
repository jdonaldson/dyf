"""Can rings/arcs be recovered from the tree's angular wedges by re-reading the norm?

The tree L2-normalises rows, so in 2-D each leaf is an angular wedge containing
points from every ring at that angle.  The norm is a 1-D coordinate the tree threw
away.  Probe: per leaf, split the sorted norms where they are bimodal (relative
separation criterion, no absolute threshold), giving (leaf, radius-band) sub-nodes;
connect sub-nodes by Euclidean kNN on their raw-space centroids; connected
components -> labels.

Outcome variable (fixed before running): ARI / AMI vs ground truth on circles,
moons, blobs (gallery settings), digits as the control that must not get worse.
Also report: leaves split, leaf purity before / after, components found.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_digits, make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors

from dyf import LazyIndex, build_dyf_tree, write_lazy_index
from dyf.agglomerate import _collect_leaf_data, louvain_cluster_leaves

SEED = 42
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)


def datasets():
    X, y = make_circles(n_samples=3000, noise=0.05, factor=0.5, random_state=SEED)
    yield "circles", X.astype(np.float32), y
    X, y = make_moons(n_samples=3000, noise=0.08, random_state=SEED)
    yield "moons", X.astype(np.float32), y
    X, y = make_blobs(
        n_samples=[200, 400, 200, 1000, 3000], cluster_std=[0.3, 0.8, 0.5, 1.5, 2.0], n_features=2, random_state=SEED
    )
    yield "blobs", X.astype(np.float32), y
    d = load_digits()
    yield "digits", d.data.astype(np.float32), d.target


def build_leaves(X):
    """Shipped tree + shipped Louvain; returns item->leaf map and shipped labels."""
    tree = build_dyf_tree(X, seed=SEED, **TREE)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "probe.dyf"
        write_lazy_index(tree, X, str(path), compression="none", quantization="float32")
        with LazyIndex(str(path)) as idx:
            shipped, *_ = louvain_cluster_leaves(idx, X[:, :2], X)
            leaves, _, leaf_point_indices, _ = _collect_leaf_data(idx)
    item_leaf = np.full(len(X), -1, dtype=np.int64)
    for li, ids in enumerate(leaf_point_indices):
        item_leaf[ids] = li
    assert (item_leaf >= 0).all()
    return item_leaf, np.asarray(shipped)


def purity(groups, y):
    tot = 0
    for g in np.unique(groups):
        m = groups == g
        tot += np.bincount(y[m]).max()
    return tot / len(y)


def split_1d(r, sep_min, min_band, depth=0, max_depth=3):
    """Recursive 2-way split of sorted 1-D values at the best Otsu cut, accepted
    only when the two bands are separated by >= sep_min pooled std.  Returns a
    band id per element (0..b-1)."""
    n = len(r)
    out = np.zeros(n, dtype=np.int64)
    if n < 2 * min_band or depth >= max_depth:
        return out
    order = np.argsort(r)
    rs = r[order]
    best, best_i = -1.0, None
    cs = np.cumsum(rs)
    cs2 = np.cumsum(rs * rs)
    for i in range(min_band, n - min_band + 1):
        n1, n2 = i, n - i
        m1 = cs[i - 1] / n1
        m2 = (cs[-1] - cs[i - 1]) / n2
        v1 = max(cs2[i - 1] / n1 - m1 * m1, 1e-12)
        v2 = max((cs2[-1] - cs2[i - 1]) / n2 - m2 * m2, 1e-12)
        sep = (m2 - m1) / np.sqrt((v1 + v2) / 2)
        if sep > best:
            best, best_i = sep, i
    if best_i is None or best < sep_min:
        return out
    lo, hi = order[:best_i], order[best_i:]
    out[lo] = split_1d(r[lo], sep_min, min_band, depth + 1, max_depth)
    out[hi] = split_1d(r[hi], sep_min, min_band, depth + 1, max_depth) + out[lo].max() + 1
    return out


def refine(X, item_leaf, sep_min, min_band):
    """(leaf, band) sub-node id per item, plus how many leaves split."""
    sub = np.zeros(len(X), dtype=np.int64)
    nxt, n_split = 0, 0
    r = np.linalg.norm(X, axis=1)
    for li in np.unique(item_leaf):
        m = np.flatnonzero(item_leaf == li)
        bands = split_1d(r[m], sep_min, min_band)
        if bands.max() > 0:
            n_split += 1
        sub[m] = bands + nxt
        nxt += bands.max() + 1
    return sub, n_split


def components(X, sub, k, mutual):
    ids = np.unique(sub)
    C = np.stack([X[sub == s].mean(0) for s in ids])
    kk = min(k + 1, len(C))
    nn = NearestNeighbors(n_neighbors=kk).fit(C)
    _, J = nn.kneighbors(C)
    rows = np.repeat(np.arange(len(C)), kk - 1)
    cols = J[:, 1:].ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(C), len(C))).tocsr()
    A = A.minimum(A.T) if mutual else A.maximum(A.T)
    ncomp, lab = connected_components(A, directed=False)
    remap = {s: i for i, s in enumerate(ids)}
    return ncomp, lab[np.vectorize(remap.get)(sub)]


def main():
    sep_min = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    min_band = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"sep_min={sep_min} min_band={min_band}  tree={TREE}")
    for name, X, y in datasets():
        t0 = time.perf_counter()
        item_leaf, shipped = build_leaves(X)
        n_leaves = len(np.unique(item_leaf))
        print(
            f"\n== {name}: n={len(X)} d={X.shape[1]} leaves={n_leaves} "
            f"leaf purity={purity(item_leaf, y):.3f}  "
            f"shipped Louvain k={len(np.unique(shipped))} ARI={ari(y, shipped):.3f} AMI={ami(y, shipped):.3f}"
        )
        sub, n_split = refine(X, item_leaf, sep_min, min_band)
        print(
            f"   wedge->bands: {n_split}/{n_leaves} leaves split, {len(np.unique(sub))} sub-nodes, "
            f"purity={purity(sub, y):.3f}"
        )
        # unrefined control: components on plain leaves (same graph rule)
        for label, groups in (("plain leaves", item_leaf), ("bands", sub)):
            for k in (3, 5):
                for mutual in (False, True):
                    nc, lab = components(X, groups, k, mutual)
                    print(
                        f"   {label:12s} k={k} {'mutual' if mutual else 'union '}: "
                        f"comps={nc:3d} ARI={ari(y, lab):.3f} AMI={ami(y, lab):.3f}"
                    )
        print(f"   ({time.perf_counter() - t0:.1f}s)")


def oracle_linkers(X, sub, n_clusters, k=5):
    """Spectral (kNN affinity) and single-linkage on sub-node centroids, told k."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import SpectralClustering

    ids = np.unique(sub)
    C = np.stack([X[sub == s].mean(0) for s in ids])
    remap = np.vectorize({s: i for i, s in enumerate(ids)}.get)(sub)
    out = {}
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=min(k, len(C) - 1),
        random_state=0,
        assign_labels="discretize",
    ).fit(C)
    out["spectral"] = sc.labels_[remap]
    Z = linkage(C, method="single")
    out["single"] = fcluster(Z, n_clusters, criterion="maxclust")[remap]
    return out


def main_oracle():
    sep_min, min_band = 3.0, 5
    for name, X, y in datasets():
        if name == "digits":
            continue
        item_leaf, _ = build_leaves(X)
        sub, _ = refine(X, item_leaf, sep_min, min_band)
        kt = len(np.unique(y))
        print(f"== {name} oracle k={kt}")
        for label, groups in (("plain leaves", item_leaf), ("bands", sub)):
            for how, lab in oracle_linkers(X, groups, kt).items():
                print(f"   {label:12s} {how:8s}: ARI={ari(y, lab):.3f} AMI={ami(y, lab):.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "oracle":
        main_oracle()
    else:
        main()
