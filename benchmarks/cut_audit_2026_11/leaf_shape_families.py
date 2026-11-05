"""Per-LEAF shape families from the path profile, and whether the same shapes recur in
different parts of the embedding.

Every point in a leaf shares its path, so the dimension profile is a property of the leaf
(branch).  Per dataset (seed 42), 8-way depth-8 tree:
  profile      per-rung local dimension along the leaf's path (rungs with child node < 64
               points masked), as in dimension_profile_test.py
  family       nearest synthetic dictionary family (ball / curve / sheet / blobs / mixture)
               by RMS distance in log(1+d) over the first 4 shared rungs (>= 2 needed)
Shipped depth-4 pipeline on the same data gives each point a final community; each deep
leaf gets: size, majority class + purity, re-cut moved fraction, majority community,
unit centroid.

1. Coherence: are family labels aligned with classes?  AMI(class-of-leaf, family), leaf-
   weighted, vs a shuffle null (family labels permuted over leaves, 20 draws).  Per family:
   leaves, points share, mean purity, moved fraction.  Per class (largest 8): family mix.
2. Recurrence: do leaves with the same shape sit together or scatter?  k-means (k=5) on
   log(1+profile) (forward-filled) over leaves with >= 3 rungs.  Per profile-cluster: how
   many final communities hold >= 5% of its points (spread), mean pairwise cosine of its
   leaf centroids vs the dataset's random-pair mean (0 = as scattered as random), class
   entropy.  Same per family label.  A shape that recurs across the embedding has many
   communities and near-random cosine.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python leaf_shape_families.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402
from dimension_profile_test import MIN_NODE, unit  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
FAMS = ["ball", "curve", "sheet", "blobs", "mixture"]
DICT = {
    k: v["mean_profile"]
    for k, v in json.loads((OUT / "dimension_profile_test_results.json").read_text())["synthetic"].items()
}


def leaf_profiles(tree, Xn):
    """Walk the deep tree; per leaf: profile over rungs, indices, unit centroid."""
    leaves = []

    def stats(idx):
        c = Xn[idx].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        return c, max(1.0 - float((Xn[idx] @ c).mean()), 1e-6)

    def walk(node, ladder):
        idx = np.asarray(node["indices"])
        c, r = stats(idx)
        lad = ladder + [(len(idx), r)]
        ch = node.get("children") or []
        if not ch:
            prof = []
            for j in range(1, len(lad)):
                n0, r0 = lad[j - 1]
                n1, r1 = lad[j]
                dr = np.log(1 / r1) - np.log(1 / r0)
                d = (np.log(n0) - np.log(n1)) / dr if dr > 1e-6 else np.nan
                prof.append(np.nan if (n1 < MIN_NODE or not np.isfinite(d)) else float(np.clip(d, 0, 60)))
            leaves.append(dict(indices=idx, profile=prof, centroid=c))
            return
        for k in ch:
            walk(k, lad)

    walk(tree, [])
    return leaves


def nearest_family(prof):
    best, bd = None, np.inf
    for f in FAMS:
        fp = DICT[f]
        pairs = [
            (a, b)
            for a, b in zip(prof[:4], fp[:4])
            if a is not None and not np.isnan(a) and b is not None and not np.isnan(b)
        ]
        if len(pairs) < 2:
            continue
        d = np.sqrt(np.mean([(np.log1p(a) - np.log1p(b)) ** 2 for a, b in pairs]))
        if d < bd:
            best, bd = f, d
    return best, bd


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    leaves = leaf_profiles(build_dyf_tree(X, seed=42, **TREE), Xn)
    # shipped pipeline
    item_leaf4 = leaf_map(build_dyf_tree(X, seed=42, **SHIPPED), n)
    L4 = item_leaf4.max() + 1
    cent4 = np.zeros((L4, X.shape[1]), dtype=np.float32)
    np.add.at(cent4, item_leaf4, Xn)
    cent4 = unit(cent4)
    ll, _, _ = _run_louvain_on_centroids(cent4, k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(ll)[item_leaf4].astype(np.int32)
    final = _reassign_points(block.copy(), X)
    moved = final != block
    # per-leaf table
    rows = []
    for lf in leaves:
        idx = lf["indices"]
        fam, dist = nearest_family(lf["profile"])
        cls = np.bincount(y[idx]).argmax()
        rows.append(
            dict(
                size=len(idx),
                fam=fam,
                dist=dist,
                cls=int(cls),
                purity=float(np.mean(y[idx] == cls)),
                moved=float(moved[idx].mean()),
                comm=int(np.bincount(final[idx]).argmax()),
                profile=lf["profile"],
                centroid=lf["centroid"],
            )
        )
    rows = [r for r in rows if r["fam"] is not None]
    size = np.array([r["size"] for r in rows])
    fam = np.array([r["fam"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    comm = np.array([r["comm"] for r in rows])
    C = np.stack([r["centroid"] for r in rows])
    out = {"leaves": len(rows), "points_covered": float(size.sum() / n)}
    print(f"\n== {name}: {len(rows)} deep leaves with a family label ({size.sum() / n:.0%} of points)", flush=True)
    # 1. coherence
    w = np.repeat(np.arange(len(rows)), size)  # leaf-weighted by expanding
    real = float(ami(cls[w], fam[w]))
    rng = np.random.default_rng(0)
    null = [float(ami(cls[w], rng.permutation(fam)[w])) for _ in range(20)]
    out["ami_class_family"] = real
    out["ami_null"] = [float(np.mean(null)), float(np.std(null))]
    print(
        f"   AMI(class, family) leaf-weighted = {real:.3f}   shuffle null {np.mean(null):.3f} ± {np.std(null):.3f}",
        flush=True,
    )
    out["families"] = {}
    print(
        "   family   leaves  points  purity  moved   communities(>=5% of its points)   mean cos within vs random",
        flush=True,
    )
    rand_cos = float(np.mean((C[rng.integers(0, len(C), 2000)] * C[rng.integers(0, len(C), 2000)]).sum(1)))
    for f in FAMS:
        m = fam == f
        if not m.any():
            continue
        pts = size[m].sum()
        cm = np.bincount(comm[m], weights=size[m])
        spread = int((cm >= 0.05 * pts).sum())
        Cm = C[m]
        if m.sum() > 1:
            ii, jj = rng.integers(0, m.sum(), 1500), rng.integers(0, m.sum(), 1500)
            keep = ii != jj
            within = float(np.mean((Cm[ii[keep]] * Cm[jj[keep]]).sum(1)))
        else:
            within = float("nan")
        row = dict(
            leaves=int(m.sum()),
            points=float(pts / n),
            purity=float(np.average([r["purity"] for r in rows if r["fam"] == f], weights=size[m])),
            moved=float(np.average([r["moved"] for r in rows if r["fam"] == f], weights=size[m])),
            communities=spread,
            within_cos=within,
            random_cos=rand_cos,
        )
        out["families"][f] = row
        print(
            f"   {f:8s} {row['leaves']:6d}  {row['points']:5.1%}   {row['purity']:.2f}   {row['moved']:.1%}   {spread:3d}                                {within:.2f} vs {rand_cos:.2f}",
            flush=True,
        )
    out["classes"] = {}
    print("   class (largest 8): family mix of its leaves (by points)", flush=True)
    for c in np.argsort(-np.bincount(y))[:8]:
        m = cls == c
        if not m.any():
            continue
        mix = {f: float(size[m & (fam == f)].sum() / size[m].sum()) for f in FAMS}
        out["classes"][int(c)] = mix
        print(f"     {int(c):4d}: " + "  ".join(f"{f} {v:.0%}" for f, v in mix.items() if v > 0), flush=True)
    # 2. recurrence via profile clusters
    P = np.array([r["profile"][:6] + [np.nan] * (6 - len(r["profile"][:6])) for r in rows], dtype=float)
    ok = (~np.isnan(P)).sum(1) >= 3
    Q = np.log1p(P[ok])
    for j in range(1, Q.shape[1]):
        bad = np.isnan(Q[:, j])
        Q[bad, j] = Q[bad, j - 1]
    Q[np.isnan(Q)] = np.nanmean(Q)
    km = KMeans(n_clusters=5, n_init=5, random_state=0).fit(Q)
    lab = km.labels_
    idx_ok = np.flatnonzero(ok)
    out["profile_clusters"] = []
    print(
        "   profile clusters (k=5 on log(1+d), leaves with >=3 rungs): points | mean profile | communities | within cos vs random | class entropy (nats) | purity",
        flush=True,
    )
    for k in range(5):
        m = lab == k
        ii = idx_ok[m]
        pts = size[ii].sum()
        cm = np.bincount(comm[ii], weights=size[ii])
        spread = int((cm >= 0.05 * pts).sum())
        Ck = C[ii]
        a, b = rng.integers(0, len(ii), 1500), rng.integers(0, len(ii), 1500)
        keep = a != b
        within = float(np.mean((Ck[a[keep]] * Ck[b[keep]]).sum(1))) if keep.any() else float("nan")
        ce = entropy(np.bincount(cls[ii], weights=size[ii]))
        prof = np.expm1(km.cluster_centers_[k])
        pur = float(np.average([rows[i]["purity"] for i in ii], weights=size[ii]))
        row = dict(
            points=float(pts / n),
            profile=[float(v) for v in prof],
            communities=spread,
            within_cos=within,
            class_entropy=ce,
            purity=pur,
        )
        out["profile_clusters"].append(row)
        print(
            f"     {pts / n:5.1%} | {' '.join(f'{v:4.1f}' for v in prof)} | {spread:3d} | {within:.2f} vs {rand_cos:.2f} | {ce:.2f} | {pur:.2f}",
            flush=True,
        )
    out["class_entropy_dataset"] = entropy(np.bincount(y).astype(float))
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "leaf_shape_families_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
