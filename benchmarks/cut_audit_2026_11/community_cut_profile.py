"""Does the density profile ACROSS a community boundary separate slices from leaks?

For each of a dataset's top-10 kNN-contact community pairs (c, d):
  u   = unit vector from centroid(c) to centroid(d) in raw Euclidean space
  t   = projection of every point of c and d onto u (origin at centroid(c))
  t*  = the threshold on t that best separates c from d (min misclassification)
  f   = smoothed histogram density of the pooled projections
Features per pair:
  valley    = f(t*) / min(peak of f left of t*, peak right of t*)
              ~1 or >1 : no valley, the cut goes through continuous density (slice)
              <<1      : a density valley at the cut (boundary)
  c_side    = f(t*) / peak left of t*   (does c's density rise toward the cut?)
  d_side    = f(t*) / peak right of t*
  slope_c   = sign of df/dt at t* approached from c's side (+ = rising toward cut)
  sep       = |mean t_d - mean t_c| / pooled std  (how far apart along u)
Score: AUC of each feature for slice (same majority label) vs not, per dataset and
pooled over all datasets.  Needs only X, labels, centroids — no kNN graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
KNN = 15
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
BINS, SMOOTH = 120, 2.0


def load(name):
    meta = json.loads((BENCH / "datasets.json").read_text())[name]
    X = np.ascontiguousarray(np.load(meta.get("x_path", BENCH / f"{name}_X.npy")), dtype=np.float32)
    y = np.load(meta.get("y_path", BENCH / f"{name}_y.npy"), allow_pickle=True)
    _, y = np.unique(y, return_inverse=True)
    return X, y


def top_pairs(lab, J, n_top=10):
    n, k = J.shape
    C = lab.max() + 1
    flow = np.zeros((C, C), dtype=np.int64)
    np.add.at(flow, (np.repeat(lab, k), lab[J.ravel()]), 1)
    out = flow.sum(1) - np.diag(flow)
    share = flow / np.maximum(out, 1)[:, None]
    np.fill_diagonal(share, 0)
    off = ~np.eye(C, dtype=bool)
    return [tuple(p) for p in np.argwhere(off)[np.argsort(-share[off])[:n_top]]], share


def profile(X, lab, c, d):
    Xc, Xd = X[lab == c], X[lab == d]
    mc, md = Xc.mean(0), Xd.mean(0)
    u = md - mc
    u /= np.linalg.norm(u) + 1e-12
    tc, td = (Xc - mc) @ u, (Xd - mc) @ u
    t = np.concatenate([tc, td])
    is_d = np.concatenate([np.zeros(len(tc), bool), np.ones(len(td), bool)])
    # best threshold: minimise misclassification (c below, d above)
    order = np.argsort(t)
    ts, isd = t[order], is_d[order]
    c_below = np.cumsum(~isd)  # c points at or below position i
    d_below = np.cumsum(isd)
    err = (len(tc) - c_below) + d_below  # c above + d below
    i = int(np.argmin(err))
    t_star = ts[i]
    lo, hi = np.percentile(t, [0.5, 99.5])
    h, edges = np.histogram(t, bins=BINS, range=(lo, hi), density=True)
    f = gaussian_filter1d(h, SMOOTH)
    centers = 0.5 * (edges[1:] + edges[:-1])
    b = int(np.clip(np.searchsorted(centers, t_star), 1, BINS - 2))
    f_cut = f[b]
    peak_l = f[:b].max() if b > 0 else f_cut
    peak_r = f[b + 1 :].max() if b < BINS - 1 else f_cut
    slope_c = float(np.sign(f[b] - f[max(b - 3, 0)]))  # rising toward the cut from c's side?
    slope_d = float(np.sign(f[b] - f[min(b + 3, BINS - 1)]))
    sep = abs(td.mean() - tc.mean()) / (np.sqrt(0.5 * (tc.var() + td.var())) + 1e-12)
    return dict(
        valley=float(f_cut / max(min(peak_l, peak_r), 1e-12)),
        c_side=float(f_cut / max(peak_l, 1e-12)),
        d_side=float(f_cut / max(peak_r, 1e-12)),
        slope_c=slope_c,
        slope_d=slope_d,
        sep=float(sep),
        both_rising=float(slope_c > 0 and slope_d > 0),
    )


def main(names):
    pooled = {"slice": [], "feats": []}
    results = {}
    for name in names:
        X, y = load(name)
        J = np.load(OUT / f"knn{KNN}_{name}.npy")
        lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
        _, lab = np.unique(lab, return_inverse=True)
        pairs, share = top_pairs(lab, J)
        maj = np.array([np.bincount(y[lab == c]).argmax() for c in range(lab.max() + 1)])
        rows = []
        print(f"\n== {name}")
        for c, d in pairs:
            p = profile(X, lab, c, d)
            p.update(c=int(c), d=int(d), share=float(share[c, d]), slice=bool(maj[c] == maj[d]))
            rows.append(p)
            pooled["slice"].append(p["slice"])
            pooled["feats"].append(p)
            print(
                f"   {c:3d}->{d:3d} share={p['share']:.2f} {'SLICE' if p['slice'] else 'leak '} "
                f"valley={p['valley']:.2f} c_side={p['c_side']:.2f} d_side={p['d_side']:.2f} "
                f"slope c{'+' if p['slope_c'] > 0 else '-'} d{'+' if p['slope_d'] > 0 else '-'} sep={p['sep']:.2f}"
            )
        s = np.array([r["slice"] for r in rows])
        aucs = {}
        if 0 < s.sum() < len(s):
            for fkey in ("valley", "c_side", "d_side", "both_rising", "sep"):
                v = np.array([r[fkey] for r in rows])
                aucs[fkey] = float(roc_auc_score(s, v))
            print("   AUC slice-vs-leak: " + "  ".join(f"{k}={v:.2f}" for k, v in aucs.items()))
        else:
            print(f"   AUC n/a ({s.sum()}/10 slices)")
        results[name] = dict(pairs=rows, auc=aucs)
    s = np.array(pooled["slice"])
    print(f"\n== pooled over {len(s)} pairs, {s.sum()} slices")
    pooled_auc = {}
    for fkey in ("valley", "c_side", "d_side", "both_rising", "sep"):
        v = np.array([r[fkey] for r in pooled["feats"]])
        pooled_auc[fkey] = float(roc_auc_score(s, v))
    print("   AUC slice-vs-leak: " + "  ".join(f"{k}={v:.2f}" for k, v in pooled_auc.items()))
    results["pooled_auc"] = pooled_auc
    (OUT / "community_cut_profile_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
