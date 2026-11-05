"""Second pass on the cut profile with an OWN-density test.

Calibration (cut_profile_calibration.py) showed the pooled 'shoulder' rule (c_side >= 0.95)
also fires for a small DISTINCT Gaussian within ~3 sigma of a 4x bigger neighbour: in the
pooled 1-D density the big tail hides the small mode.  Fix: look at c's OWN projected
density.  A flank shaved off d peaks AT the cut (truncated); a distinct small blob peaks in
its interior.
  own_offset = (t_cut - argmax of c's own density) / std(t_c)    ~0 = peak at the cut
Strict shoulder = pooled no-valley (c_side >= 0.95) AND own_offset < 0.5.
Runs dyf (defaults), k-means@k, Leiden@closest-k over all contact pairs (share >= 0.05).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_cut_profile_full import BENCH, BINS, KNN, OUT, SMOOTH, KMeans, contact_pairs, load  # noqa: E402

ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def profile2(X, lab, c, d):
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
    hc, _ = np.histogram(tc, bins=edges, density=True)
    fc = gaussian_filter1d(hc, SMOOTH)
    own_offset = (t_star - centers[int(np.argmax(fc))]) / (tc.std() + 1e-12)
    return dict(
        valley=float(f_cut / max(min(peak_l, peak_r), 1e-12)),
        c_side=float(f_cut / max(peak_l, 1e-12)),
        d_side=float(f_cut / max(peak_r, 1e-12)),
        own_offset=float(own_offset),
    )


def summarise2(X, lab, pairs):
    vals = [profile2(X, lab, c, d) for c, d, _ in pairs]
    if not vals:
        return dict(n_pairs=0, k=int(lab.max() + 1))
    valley = np.array([v["valley"] for v in vals])
    c_side = np.array([v["c_side"] for v in vals])
    off = np.array([v["own_offset"] for v in vals])
    pooled = c_side >= 0.95
    strict = pooled & (off < 0.5)
    strict_comms = {pairs[i][0] for i in np.flatnonzero(strict)}
    return dict(
        n_pairs=len(vals),
        k=int(lab.max() + 1),
        valley_lt03=float((valley < 0.3).mean()),
        shoulder_pooled=float(pooled.mean()),
        shoulder_strict=float(strict.mean()),
        shoulder_strict_pts=float(np.isin(lab, list(strict_comms)).mean()) if strict_comms else 0.0,
        hidden_small_blob=float((pooled & (off >= 0.5)).mean()),
        own_offset_median=float(np.median(off)),
    )


def fmt(tag, r):
    if r["n_pairs"] == 0:
        return f"   {tag:12s} k={r['k']:4d} no contact pairs"
    return (
        f"   {tag:12s} k={r['k']:4d} pairs={r['n_pairs']:4d} valley<.3={r['valley_lt03']:.2f} "
        f"shoulder pooled={r['shoulder_pooled']:.2f} strict={r['shoulder_strict']:.2f} "
        f"(pts {r['shoulder_strict_pts']:.1%}) hidden-small-blob={r['hidden_small_blob']:.2f} "
        f"med own_offset={r['own_offset_median']:.2f}"
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
    out = {}
    for tag, lab in parts.items():
        out[tag] = summarise2(X, lab, contact_pairs(lab, J))
        print(fmt(tag, out[tag]), flush=True)
    out["seconds"] = time.perf_counter() - t0
    print(f"   ({out['seconds']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "community_cut_profile_ownmode_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
