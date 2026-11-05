"""Bigger test of the cut-profile typology (community_cut_profile.py):
  * ALL ordered community pairs with kNN contact share >= 0.05, not the top 10
  * three partitions on the same data: dyf shipped path (defaults), k-means told dyf's k,
    pynndescent+Leiden at the closest k  -> is the shoulder count a dyf artefact?
  * dyf rebuilt with seeds 1 and 2 through the shipped path -> is the count stable?
  * threshold sensitivity: shoulder at c_side >= {0.90, 0.95, 1.00}, valley at {0.2, 0.3, 0.4}
Outputs per (dataset, partition): n pairs, fraction deep-valley / shoulder / saddle at each
threshold, share of POINTS sitting in a shoulder community, and the median valley.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

from dyf import LazyIndex, build_dyf_tree, write_lazy_index
from dyf.agglomerate import louvain_cluster_leaves

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
KNN, SHARE_MIN = 15, 0.05
BINS, SMOOTH = 120, 2.0
SHOULDER_T, VALLEY_T = (0.90, 0.95, 1.00), (0.2, 0.3, 0.4)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def load(name):
    meta = json.loads((BENCH / "datasets.json").read_text())[name]
    X = np.ascontiguousarray(np.load(meta.get("x_path", BENCH / f"{name}_X.npy")), dtype=np.float32)
    return X


def dyf_shipped(X, seed):
    tree = build_dyf_tree(X, seed=seed, **TREE)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "p.dyf")
        write_lazy_index(tree, X, path, compression="none", quantization="float32")
        with LazyIndex(path) as idx:
            lab, *_ = louvain_cluster_leaves(idx, X[:, :2], X)
    return np.asarray(lab)


def contact_pairs(lab, J):
    n, k = J.shape
    C = lab.max() + 1
    flow = np.zeros((C, C), dtype=np.int64)
    np.add.at(flow, (np.repeat(lab, k), lab[J.ravel()]), 1)
    out = flow.sum(1) - np.diag(flow)
    share = flow / np.maximum(out, 1)[:, None]
    np.fill_diagonal(share, 0)
    return [(int(c), int(d), float(share[c, d])) for c, d in np.argwhere(share >= SHARE_MIN)]


def profile(X, lab, c, d):
    Xc, Xd = X[lab == c], X[lab == d]
    mc, md = Xc.mean(0), Xd.mean(0)
    u = md - mc
    u /= np.linalg.norm(u) + 1e-12
    tc, td = (Xc - mc) @ u, (Xd - mc) @ u
    t = np.concatenate([tc, td])
    is_d = np.concatenate([np.zeros(len(tc), bool), np.ones(len(td), bool)])
    order = np.argsort(t)
    ts, isd = t[order], is_d[order]
    err = (len(tc) - np.cumsum(~isd)) + np.cumsum(isd)
    t_star = ts[int(np.argmin(err))]
    lo, hi = np.percentile(t, [0.5, 99.5])
    h, edges = np.histogram(t, bins=BINS, range=(lo, hi), density=True)
    f = gaussian_filter1d(h, SMOOTH)
    centers = 0.5 * (edges[1:] + edges[:-1])
    b = int(np.clip(np.searchsorted(centers, t_star), 1, BINS - 2))
    f_cut = f[b]
    peak_l, peak_r = f[:b].max(), f[b + 1 :].max()
    return float(f_cut / max(min(peak_l, peak_r), 1e-12)), float(f_cut / max(peak_l, 1e-12))


def summarise(X, lab, pairs):
    len(X)
    vals = [(c, d, s, *profile(X, lab, c, d)) for c, d, s in pairs]
    valley = np.array([v[3] for v in vals])
    c_side = np.array([v[4] for v in vals])
    res = dict(n_pairs=len(vals), k=int(lab.max() + 1), valley_median=float(np.median(valley)) if len(vals) else None)
    for st in SHOULDER_T:
        sh = c_side >= st
        comms = {v[0] for v, m in zip(vals, sh) if m}
        res[f"shoulder@{st}"] = float(sh.mean()) if len(vals) else None
        res[f"shoulder_pts@{st}"] = float(np.isin(lab, list(comms)).mean()) if comms else 0.0
    for vt in VALLEY_T:
        res[f"valley@{vt}"] = float((valley < vt).mean()) if len(vals) else None
    res["saddle"] = float(((valley >= 0.3) & (c_side < 0.95)).mean()) if len(vals) else None
    return res, vals


def fmt(tag, r):
    return (
        f"   {tag:12s} k={r['k']:4d} pairs={r['n_pairs']:4d} valley<.3={r['valley@0.3']:.2f} "
        f"saddle={r['saddle']:.2f} shoulder@.95={r['shoulder@0.95']:.2f} "
        f"(pts {r['shoulder_pts@0.95']:.1%}; @.90 {r['shoulder@0.9']:.2f}, @1.0 {r['shoulder@1.0']:.2f}) "
        f"med valley={r['valley_median']:.2f}"
    )


def run(name):
    t0 = time.perf_counter()
    X = load(name)
    J = np.load(OUT / f"knn{KNN}_{name}.npy")
    print(f"\n== {name}: n={len(X)}", flush=True)
    parts = {}
    lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
    _, parts["dyf"] = np.unique(lab, return_inverse=True)
    kd = parts["dyf"].max() + 1
    parts["kmeans@k"] = KMeans(n_clusters=kd, n_init=1, random_state=42).fit_predict(X)
    best, best_lab = None, None
    for r in ("0.5", "1.0", "2.0", "4.0"):
        l = np.load(BENCH / f"sB_labels_{name}_pynn_leiden_res{r}.npy")
        _, l = np.unique(l, return_inverse=True)
        if best is None or abs(l.max() + 1 - kd) < abs(best_lab.max() + 1 - kd):
            best, best_lab = r, l
    parts[f"leiden@{best}"] = best_lab
    for seed in (1, 2):
        _, parts[f"dyf_seed{seed}"] = np.unique(dyf_shipped(X, seed), return_inverse=True)
    out = {}
    for tag, lab in parts.items():
        r, vals = summarise(X, lab, contact_pairs(lab, J))
        out[tag] = r
        print(fmt(tag, r), flush=True)
    out["seconds"] = time.perf_counter() - t0
    print(f"   ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "community_cut_profile_full_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
