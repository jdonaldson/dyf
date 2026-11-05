"""Cluster leaves on their shape metrics: do the clusters follow the tree's branching, do the
same shapes recur on different branches, and are those branches related?

Per dataset (seed 42), 8-way depth-8 tree.  Per leaf:
  profile    per-rung local dimension (first 6 rungs, masked below 64 points, forward-filled)
  summary    rung-free shape metrics: mean d, slope over rungs, max d, min d, largest radius
             step, leaf radius, log leaf size, depth
  ancestry   depth-1 / depth-2 / depth-3 ancestor ids
Two clusterings (k-means, k=6, standardised, log1p on dimensions): on the profile, and on the
rung-free summary.  For each:
  branching dominance   AMI(shape cluster, depth-d ancestor) for d = 1, 2, 3, vs a null that
                        permutes cluster labels over leaves (20 draws); and the share of each
                        cluster's leaves inside its largest depth-1 subtree
  recurrence            depth-1 subtrees holding >= 5% of the cluster's leaves (of 8)
  relatedness           among leaf pairs that share a shape cluster but sit in DIFFERENT
                        depth-1 subtrees: same majority class rate, same final community rate,
                        centroid cosine — each vs random cross-subtree pairs (null)
Outcome fixed before running: if AMI(shape, depth-1) is near its null, branching does not
dominate; if cross-branch same-shape pairs share class / community / closeness above the
null, the recurring shapes mark related regions rather than coincidence.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python shape_clusters_vs_branches.py [names]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import StandardScaler

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402
from dimension_profile_test import MIN_NODE, unit  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=8, min_leaf_size=20)
SHIPPED = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
KC, NR, NPAIR = 6, 6, 20000


def leaves_with_ancestry(tree, Xn):
    rows = []

    def stats(idx):
        c = Xn[idx].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        return c, max(1.0 - float((Xn[idx] @ c).mean()), 1e-6)

    def walk(node, lad, path):
        idx = np.asarray(node["indices"])
        c, r = stats(idx)
        lad = lad + [(len(idx), r)]
        ch = node.get("children") or []
        if not ch:
            prof, steps = [], []
            for j in range(1, len(lad)):
                n0, r0 = lad[j - 1]
                n1, r1 = lad[j]
                dr = np.log(1 / r1) - np.log(1 / r0)
                d = (np.log(n0) - np.log(n1)) / dr if dr > 1e-6 else np.nan
                prof.append(np.nan if (n1 < MIN_NODE or not np.isfinite(d)) else float(np.clip(d, 0, 60)))
                steps.append(np.log(r0) - np.log(r1))
            rows.append(
                dict(
                    indices=idx,
                    profile=prof,
                    steps=steps,
                    radius=r,
                    size=len(idx),
                    depth=len(path),
                    path=tuple(path),
                    centroid=c,
                )
            )
            return
        for j, k in enumerate(ch):
            walk(k, lad, path + [j])

    walk(tree, [], [])
    return rows


def features(rows):
    P = np.full((len(rows), NR), np.nan)
    S = np.zeros((len(rows), 8))
    for i, r in enumerate(rows):
        pr = np.array((r["profile"] + [np.nan] * NR)[:NR], dtype=float)
        P[i] = pr
        ok = ~np.isnan(pr)
        v = pr[ok]
        x = np.flatnonzero(ok)
        slope = np.polyfit(x, v, 1)[0] if len(v) >= 2 and np.ptp(x) > 0 else 0.0
        S[i] = [
            np.log1p(v.mean()) if len(v) else 0,
            slope,
            np.log1p(v.max()) if len(v) else 0,
            np.log1p(v.min()) if len(v) else 0,
            max(r["steps"]) if r["steps"] else 0,
            np.log(r["radius"]),
            np.log(r["size"]),
            r["depth"],
        ]
    Q = np.log1p(P)
    for j in range(1, NR):
        bad = np.isnan(Q[:, j])
        Q[bad, j] = Q[bad, j - 1]
    Q[np.isnan(Q)] = np.nanmean(Q)
    return Q, S, P


def anc(rows, d):
    return np.array([hash(r["path"][:d]) % (10**9) for r in rows])


def cross_branch_pairs(lab, a1, rng, n):
    """pairs (i, j) with same shape cluster and different depth-1 ancestor; and random pairs with different ancestor."""
    same, rand = [], []
    L = len(lab)
    tries = 0
    while len(same) < n and tries < 40 * n:
        i, j = rng.integers(0, L, 2)
        tries += 1
        if a1[i] == a1[j] or i == j:
            continue
        if lab[i] == lab[j]:
            same.append((i, j))
    while len(rand) < n:
        i, j = rng.integers(0, L, 2)
        if a1[i] != a1[j]:
            rand.append((i, j))
    return np.array(same), np.array(rand)


def run(name):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    rows = leaves_with_ancestry(build_dyf_tree(X, seed=42, **TREE), Xn)
    Q, S, P = features(rows)
    size = np.array([r["size"] for r in rows])
    C = np.stack([r["centroid"] for r in rows])
    cls = np.array([np.bincount(y[r["indices"]]).max() and np.bincount(y[r["indices"]]).argmax() for r in rows])
    # final communities from the shipped pipeline
    item_leaf = leaf_map(build_dyf_tree(X, seed=42, **SHIPPED), n)
    L4 = item_leaf.max() + 1
    cent = np.zeros((L4, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    ll, _, _ = _run_louvain_on_centroids(unit(cent), k=10, resolution=1.0, similarity_threshold=0.5)
    final = _reassign_points(np.asarray(ll)[item_leaf].astype(np.int32), X)
    comm = np.array([np.bincount(final[r["indices"]]).argmax() for r in rows])
    A = {d: anc(rows, d) for d in (1, 2, 3)}
    rng = np.random.default_rng(0)
    out = {"leaves": len(rows)}
    print(
        f"\n== {name}: {len(rows)} leaves, depth-1 subtrees {len(np.unique(A[1]))}, depth-2 {len(np.unique(A[2]))}",
        flush=True,
    )
    for fname, F in (("profile", Q), ("summary", S)):
        lab = KMeans(n_clusters=KC, n_init=5, random_state=0).fit_predict(StandardScaler().fit_transform(F))
        res = {"ami_ancestry": {}, "clusters": []}
        line = f"   [{fname}] AMI(shape, ancestor) "
        for d in (1, 2, 3):
            real = float(ami(lab, A[d]))
            null = [float(ami(rng.permutation(lab), A[d])) for _ in range(20)]
            res["ami_ancestry"][d] = dict(real=real, null=float(np.mean(null)))
            line += f"d{d}: {real:.3f} (null {np.mean(null):.3f})  "
        print(line, flush=True)
        same, rand = cross_branch_pairs(lab, A[1], rng, NPAIR)

        def rate(pairs, f):
            return float(np.mean([f(i, j) for i, j in pairs])) if len(pairs) else float("nan")

        rel = dict(
            same_class=rate(same, lambda i, j: cls[i] == cls[j]),
            same_class_null=rate(rand, lambda i, j: cls[i] == cls[j]),
            same_comm=rate(same, lambda i, j: comm[i] == comm[j]),
            same_comm_null=rate(rand, lambda i, j: comm[i] == comm[j]),
            cos=rate(same, lambda i, j: float(C[i] @ C[j])),
            cos_null=rate(rand, lambda i, j: float(C[i] @ C[j])),
        )
        res["cross_branch"] = rel
        print(
            f"   [{fname}] cross-branch same-shape pairs vs random cross-branch pairs: same class {rel['same_class']:.3f} vs {rel['same_class_null']:.3f} | "
            f"same community {rel['same_comm']:.3f} vs {rel['same_comm_null']:.3f} | centroid cos {rel['cos']:.3f} vs {rel['cos_null']:.3f}",
            flush=True,
        )
        print(
            f"   [{fname}] cluster: leaves | points | mean profile | depth-1 subtrees >=5% | largest-subtree share | mean depth",
            flush=True,
        )
        for c in range(KC):
            m = lab == c
            pts = size[m].sum() / n
            sub = np.bincount(np.unique(A[1][m], return_inverse=True)[1], weights=size[m])
            spread = int((sub >= 0.05 * sub.sum()).sum())
            largest = float(sub.max() / sub.sum())
            prof = np.nanmean(P[m], axis=0)
            md = float(np.mean([rows[i]["depth"] for i in np.flatnonzero(m)]))
            res["clusters"].append(
                dict(
                    leaves=int(m.sum()),
                    points=float(pts),
                    profile=[None if np.isnan(v) else float(v) for v in prof],
                    subtrees=spread,
                    largest_share=largest,
                    mean_depth=md,
                )
            )
            print(
                f"     {m.sum():6d} | {pts:5.1%} | {' '.join(f'{v:4.1f}' if np.isfinite(v) else '   .' for v in prof)} | {spread} | {largest:.2f} | {md:.1f}",
                flush=True,
            )
        out[fname] = res
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ORDER
    out_path = OUT / "shape_clusters_vs_branches_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        results[name] = run(name)
        out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
