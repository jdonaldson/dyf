"""Per-point, multi-scale "continuousness" from the tree path (Justin, 2026-11-06).

A point's path root -> leaf is a ladder of scales: at each depth a node with size n_d,
centroid, angular radius r_d = 1 - mean cos(points, centroid).  Two per-point measures from
a DEEP tree (depth 8, so the ladder has enough rungs):
  dim      box-counting slope: least-squares slope of log n_d vs log(1/r_d) along the path
           (needs >= 3 rungs with r_d > 0)  -- local intrinsic dimension
  margin   at each internal node on the path, cos(x, own child centroid) - max cos(x, other
           child centroids), divided by the node's radius; summarised as the MIN over rungs
           (how close the point ever came to a boundary, in units of the local scale) and
           the MEAN.  Continuum points sit between buckets at every scale (small margins).
  step     max drop of log r_d in one level along the path (a cluster separating = a step)
Outcomes, computed from the SHIPPED depth-4 pipeline on the same data:
  moved      label-free: point's community changed under _reassign_points (block -> final)
  disagree   labelled: point's label != its final community's majority label
  leaf_pur   labelled, graded: purity of the point's depth-4 leaf
Scores: AUC of each measure for moved / disagree, Spearman with leaf purity, and dataset-level
means (which datasets look most continuous).  Seed 42 only (per-point n is large).
Pre-registered: a measure is useful if AUC(moved) >= 0.65 on most datasets, label-free.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python point_continuousness_probe.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
DEEP = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
CHUNK = 200_000


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def node_stats(Xn, idx):
    c = Xn[idx].mean(0)
    c /= np.linalg.norm(c) + 1e-12
    r = 1.0 - float((Xn[idx] @ c).mean())
    return c, max(r, 1e-6)


def ladder(tree, Xn, n, max_depth):
    """Per point: log n_d, log(1/r_d) per depth, margin ratio per internal depth."""
    logn = np.full((n, max_depth + 1), np.nan)
    logir = np.full((n, max_depth + 1), np.nan)
    mratio = np.full((n, max_depth), np.nan)

    def walk(node, depth):
        idx = np.asarray(node["indices"])
        c, r = node_stats(Xn, idx)
        logn[idx, depth] = np.log(len(idx))
        logir[idx, depth] = np.log(1.0 / r)
        ch = node.get("children") or []
        if not ch:
            return
        kid_idx = [np.asarray(k["indices"]) for k in ch]
        kid_c = np.stack([node_stats(Xn, ki)[0] for ki in kid_idx])
        own = np.empty(len(idx), dtype=np.int64)
        pos = {int(g): p for p, g in enumerate(idx)}
        for j, ki in enumerate(kid_idx):
            own[[pos[int(g)] for g in ki]] = j
        for s in range(0, len(idx), CHUNK):
            sl = idx[s : s + CHUNK]
            S = Xn[sl] @ kid_c.T
            o = own[s : s + CHUNK]
            own_sim = S[np.arange(len(sl)), o]
            S[np.arange(len(sl)), o] = -np.inf
            best_other = S.max(1) if S.shape[1] > 1 else np.full(len(sl), -1.0)
            mratio[sl, depth] = (own_sim - best_other) / r
        for k in ch:
            walk(k, depth + 1)

    walk(tree, 0)
    return logn, logir, mratio


def measures(logn, logir, mratio):
    n = logn.shape[0]
    dim = np.full(n, np.nan)
    ok = ~np.isnan(logn)
    rungs = ok.sum(1)
    for i in np.flatnonzero(rungs >= 3):
        x, y_ = logir[i, ok[i]], logn[i, ok[i]]
        if np.ptp(x) > 1e-6:
            # n ~ r^d  =>  log n = -d * log(1/r) + c  =>  d = -slope
            dim[i] = -np.polyfit(x, y_, 1)[0]
    m_min = np.nanmin(mratio, axis=1)
    m_mean = np.nanmean(mratio, axis=1)
    logr = -logir
    step = np.nanmax(-np.diff(logr, axis=1), axis=1)  # largest one-level drop in log r
    return dict(dim=dim, m_min=m_min, m_mean=m_mean, step=step, rungs=rungs)


def auc_safe(y, s):
    m = ~np.isnan(s)
    if m.sum() < 100 or y[m].sum() == 0 or y[m].sum() == m.sum():
        return None
    return float(roc_auc_score(y[m], s[m]))


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    deep = build_dyf_tree(X, seed=42, **DEEP)
    logn, logir, mratio = ladder(deep, Xn, n, DEEP["max_depth"])
    M = measures(logn, logir, mratio)
    # shipped pipeline outcomes
    item_leaf = leaf_map(build_dyf_tree(X, seed=42, **SHIPPED), n)
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    leaf_lab, _, _ = _run_louvain_on_centroids(cent, k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(leaf_lab)[item_leaf].astype(np.int32)
    final = _reassign_points(block.copy(), X)
    moved = block != final
    maj = np.array([np.bincount(y[final == c]).argmax() if (final == c).any() else -1 for c in range(final.max() + 1)])
    disagree = y != maj[final]
    leaf_pur = np.array([np.bincount(y[item_leaf == l]).max() / (item_leaf == l).sum() for l in range(L)])[item_leaf]
    out = {
        "n": n,
        "mean_rungs": float(M["rungs"].mean()),
        "moved_frac": float(moved.mean()),
        "disagree_frac": float(disagree.mean()),
    }
    print(
        f"\n== {name}: n={n} mean rungs={out['mean_rungs']:.1f} moved={moved.mean():.1%} disagree={disagree.mean():.1%}",
        flush=True,
    )
    for key, sign, label in (
        ("dim", +1, "dim (higher = more continuous?)"),
        ("m_min", -1, "-min margin ratio"),
        ("m_mean", -1, "-mean margin ratio"),
        ("step", -1, "-max step (smaller step = continuous)"),
    ):
        s = sign * M[key]
        row = dict(
            mean=float(np.nanmean(M[key])),
            median=float(np.nanmedian(M[key])),
            auc_moved=auc_safe(moved, s),
            auc_disagree=auc_safe(disagree, s),
            rho_leaf_purity=float(spearmanr(M[key], leaf_pur, nan_policy="omit").correlation),
        )
        out[key] = row
        print(
            f"   {label:38s} mean={row['mean']:.3f} | AUC moved={row['auc_moved'] if row['auc_moved'] is None else round(row['auc_moved'], 3)} "
            f"disagree={row['auc_disagree'] if row['auc_disagree'] is None else round(row['auc_disagree'], 3)} | rho(leaf purity)={row['rho_leaf_purity']:+.2f}",
            flush=True,
        )
    out["seconds"] = time.perf_counter() - t0
    print(f"   ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "point_continuousness_probe_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
