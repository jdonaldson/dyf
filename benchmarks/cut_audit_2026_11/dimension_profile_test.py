"""Dimension as a function of scale, per point, from the tree path -- and whether the
profile SHAPES are interpretable.

Per point, along its path (8-way, depth 8, min leaf 20): rung j has node size n_j and
angular radius r_j; the per-rung local dimension is
    d_j = (log n_{j-1} - log n_j) / (log(1/r_j) - log(1/r_{j-1}))
(clipped to [0, 60]; undefined where the radius did not shrink).  A profile is d_1..d_J.

A. Dictionary on known geometry (n=20k, ambient 50, structure placed on the sphere by adding
   a large offset before normalisation; isotropic noise sigma relative to structure scale):
     ball      isotropic Gaussian                      -> flat, high
     curve     1-D smooth curve + noise                -> ~1 at coarse rungs, rising at fine
     sheet     2-D smooth sheet + noise                -> ~2 then rising
     blobs     5 well-separated Gaussians              -> a spike (cluster separating) then flat
     mixture   half curve, half blobs
B. Real data (seed 42): mean profile per dataset; mean profile per class (largest 8 classes);
   k-means (k=4) on profiles over ALL points with >= 4 rungs; per profile-cluster: mean
   profile, share of points, majority-label share, re-cut moved fraction, mean leaf purity.
   Shape features per point: mean d, slope of d over rungs, depth of the largest radius
   contraction; AUC of slope for `moved`, Spearman with leaf purity.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python dimension_profile_test.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402
from point_continuousness_probe import ladder, unit  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
NSYN, DSYN, NOISE = 20000, 50, 0.05
MIN_NODE = 64


def profiles_from_ladder(logn, logir):
    """per-point per-rung dimension d_j (NaN where undefined)."""
    dn = -np.diff(logn, axis=1)  # log n_{j-1} - log n_j  (>0)
    dr = np.diff(logir, axis=1)  # log(1/r_j) - log(1/r_{j-1})
    with np.errstate(divide="ignore", invalid="ignore"):
        d = dn / dr
    # a rung is reliable only if the child node still has enough points for its radius to
    # mean anything: mask rungs whose child node has < MIN_NODE points
    small = logn[:, 1:] < np.log(MIN_NODE)
    d[(dr <= 1e-6) | ~np.isfinite(d) | small] = np.nan
    return np.clip(d, 0, 60)


def shape_features(P):
    n, J = P.shape
    mean = np.nanmean(P, axis=1)
    slope = np.full(n, np.nan)
    rungs = (~np.isnan(P)).sum(1)
    x = np.arange(J)
    for i in np.flatnonzero(rungs >= 3):
        m = ~np.isnan(P[i])
        slope[i] = np.polyfit(x[m], P[i, m], 1)[0]
    return mean, slope, rungs


def fmt_profile(p, J=7):
    return " ".join(f"{v:4.1f}" if np.isfinite(v) else "   ." for v in p[:J])


def profile_of(X):
    Xn = unit(X)
    tree = build_dyf_tree(X.astype(np.float32), seed=42, **TREE)
    logn, logir, _ = ladder(tree, Xn, len(X), TREE["max_depth"])
    return profiles_from_ladder(logn, logir)


def synthetic():
    rng = np.random.default_rng(0)
    offset = np.zeros(DSYN)
    offset[0] = 10.0
    fams = {}
    fams["ball"] = rng.normal(size=(NSYN, DSYN))
    t = rng.uniform(0, 1, NSYN)
    freqs = rng.integers(1, 4, size=(DSYN, 3))
    phases = rng.uniform(0, 2 * np.pi, size=(DSYN, 3))
    curve = np.stack(
        [np.sum(np.sin(2 * np.pi * freqs[k] * t[:, None] + phases[k]), axis=1) for k in range(DSYN)], axis=1
    )
    fams["curve"] = curve + NOISE * np.sqrt(DSYN) * rng.normal(size=(NSYN, DSYN)) / np.sqrt(DSYN) * 3
    u, v = rng.uniform(0, 1, NSYN), rng.uniform(0, 1, NSYN)
    f2 = rng.integers(1, 3, size=(DSYN, 2, 2))
    sheet = np.stack(
        [
            np.sin(2 * np.pi * f2[k, 0, 0] * u + 2 * np.pi * f2[k, 0, 1] * v)
            + np.cos(2 * np.pi * f2[k, 1, 0] * u - 2 * np.pi * f2[k, 1, 1] * v)
            for k in range(DSYN)
        ],
        axis=1,
    )
    fams["sheet"] = sheet + NOISE * 3 * rng.normal(size=(NSYN, DSYN))
    centres = rng.normal(size=(5, DSYN)) * 4.0
    lab = rng.integers(0, 5, NSYN)
    fams["blobs"] = centres[lab] + rng.normal(size=(NSYN, DSYN)) * 0.6
    fams["mixture"] = np.vstack([fams["curve"][: NSYN // 2], fams["blobs"][NSYN // 2 :]])
    print("\n=== A. synthetic dictionary: mean per-rung dimension (rung 1..7), then shape stats")
    out = {}
    for name, X in fams.items():
        X = X + offset  # place on the sphere away from the origin
        P = profile_of(X)
        mean, slope, rungs = shape_features(P)
        row = dict(
            mean_profile=[float(v) for v in np.nanmean(P, axis=0)],
            mean_dim=float(np.nanmean(mean)),
            mean_slope=float(np.nanmean(slope)),
            mean_rungs=float(rungs.mean()),
        )
        out[name] = row
        print(
            f"   {name:8s} rungs={row['mean_rungs']:.1f} | {fmt_profile(np.nanmean(P, axis=0))} | mean d={row['mean_dim']:.1f} slope/rung={row['mean_slope']:+.2f}",
            flush=True,
        )
    return out


def real(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    P = profile_of(X)
    mean, slope, rungs = shape_features(P)
    item_leaf = leaf_map(build_dyf_tree(X, seed=42, **SHIPPED), n)
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    ll, _, _ = _run_louvain_on_centroids(cent, k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(ll)[item_leaf].astype(np.int32)
    moved = _reassign_points(block.copy(), X) != block
    leaf_pur = np.array([np.bincount(y[item_leaf == l]).max() / (item_leaf == l).sum() for l in range(L)])[item_leaf]
    out = {"mean_profile": [float(v) for v in np.nanmean(P, axis=0)], "mean_rungs": float(rungs.mean())}
    print(
        f"\n== {name}: n={n} rungs={rungs.mean():.1f} | dataset mean profile: {fmt_profile(np.nanmean(P, axis=0))}",
        flush=True,
    )
    ok = ~np.isnan(slope)
    out["slope_auc_moved"] = float(roc_auc_score(moved[ok], slope[ok])) if 0 < moved[ok].sum() < ok.sum() else None
    out["slope_rho_purity"] = float(spearmanr(slope[ok], leaf_pur[ok]).correlation)
    out["mean_auc_moved"] = float(roc_auc_score(moved[ok], mean[ok]))
    print(
        f"   slope (rising = noisy manifold): AUC moved={out['slope_auc_moved']:.3f} rho(purity)={out['slope_rho_purity']:+.2f}; "
        f"mean d: AUC moved={out['mean_auc_moved']:.3f}",
        flush=True,
    )
    # per class
    classes = np.argsort(-np.bincount(y))[:8]
    out["classes"] = {}
    print("   per class (largest 8): label | share | mean profile | slope", flush=True)
    for c in classes:
        m = y == c
        pc = np.nanmean(P[m], axis=0)
        out["classes"][int(c)] = dict(
            share=float(m.mean()), profile=[float(v) for v in pc], slope=float(np.nanmean(slope[m]))
        )
        print(f"     {int(c):4d} | {m.mean():5.1%} | {fmt_profile(pc)} | {np.nanmean(slope[m]):+.2f}", flush=True)
    # profile clusters over all points with >= 4 rungs
    sel = rungs >= 4
    Q = P[sel][:, :6].copy()
    for j in range(Q.shape[1]):  # forward-fill missing rungs
        if j > 0:
            bad = np.isnan(Q[:, j])
            Q[bad, j] = Q[bad, j - 1]
    Q[np.isnan(Q)] = np.nanmean(Q)
    km = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=8192, n_init=3).fit(Q)
    lab = km.labels_
    out["profile_clusters"] = []
    print(
        "   profile clusters (k=4 over all points with >=4 rungs): share | mean profile | majority-label share | moved | leaf purity",
        flush=True,
    )
    idx = np.flatnonzero(sel)
    for c in range(4):
        mm = lab == c
        ii = idx[mm]
        maj = np.bincount(y[ii]).max() / len(ii)
        row = dict(
            share=float(mm.mean()),
            profile=[float(v) for v in Q[mm].mean(0)],
            majority_share=float(maj),
            moved=float(moved[ii].mean()),
            leaf_purity=float(leaf_pur[ii].mean()),
        )
        out["profile_clusters"].append(row)
        print(
            f"     {mm.mean():5.1%} | {fmt_profile(Q[mm].mean(0), 6)} | maj {maj:.2f} | moved {moved[ii].mean():.1%} | purity {leaf_pur[ii].mean():.2f}",
            flush=True,
        )
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "dimension_profile_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["synthetic"] = synthetic()
    out_path.write_text(json.dumps(results, indent=1))
    for name in names:
        results[name] = real(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
