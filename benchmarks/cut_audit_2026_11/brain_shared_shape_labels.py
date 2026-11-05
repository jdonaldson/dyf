"""Why do brain branches that share a shape also share a label more than random?

Brain only (seed 42, 8-way depth-8 tree, rung-free summary metrics, k=6 as in
shape_clusters_vs_branches.py).  Questions:
  1. Are scanpy types split across top-level (depth-1) subtrees?  Per type: number of depth-1
     subtrees holding >= 5% of its points, and the share in its largest subtree.
  2. Which shape cluster carries the cross-subtree same-class enrichment?  Per cluster: the
     same-class rate of its cross-subtree pairs vs random cross-subtree pairs.
  3. Which types dominate the enriched pairs, and what do those types look like (profile,
     family mix from leaf_shape_families)?
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python brain_shared_shape_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dyf import build_dyf_tree

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import load  # noqa: E402
from dimension_profile_test import unit  # noqa: E402
from shape_clusters_vs_branches import KC, TREE, anc, features, leaves_with_ancestry  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
X, y = load("brain_1m")
Xn = unit(X)
rows = leaves_with_ancestry(build_dyf_tree(X, seed=42, **TREE), Xn)
Q, S, P = features(rows)
size = np.array([r["size"] for r in rows])
cls = np.array([np.bincount(y[r["indices"]]).argmax() for r in rows])
A1 = anc(rows, 1)
subtree_ids = {a: i for i, a in enumerate(sorted(set(A1)))}
a1 = np.array([subtree_ids[a] for a in A1])
lab = KMeans(n_clusters=KC, n_init=5, random_state=0).fit_predict(StandardScaler().fit_transform(S))
rng = np.random.default_rng(0)
n_types = int(y.max() + 1)
type_share = np.bincount(y) / len(y)

print("1. scanpy types across the 8 top-level subtrees (by points), largest 12 types")
print("   type  share   subtrees>=5%   largest-subtree share   depth-1 distribution")
split = {}
for t in np.argsort(-np.bincount(y))[:12]:
    m = cls == t
    dist = np.bincount(a1[m], weights=size[m], minlength=8)
    dist = dist / dist.sum() if dist.sum() else dist
    split[int(t)] = dict(subtrees=int((dist >= 0.05).sum()), largest=float(dist.max()))
    print(
        f"   {int(t):4d}  {type_share[t]:5.1%}        {int((dist >= 0.05).sum())}             {dist.max():.2f}          {' '.join(f'{v:.2f}' for v in dist)}"
    )

print(
    "\n2. per shape cluster: cross-subtree same-class rate (pairs within the cluster, different subtree) vs random cross-subtree pairs"
)


def pairs_same_cluster(c, n=20000):
    idx = np.flatnonzero(lab == c)
    out = []
    tries = 0
    while len(out) < n and tries < 60 * n:
        i, j = idx[rng.integers(0, len(idx), 2)]
        tries += 1
        if i != j and a1[i] != a1[j]:
            out.append((i, j))
    return np.array(out)


rand = []
while len(rand) < 20000:
    i, j = rng.integers(0, len(rows), 2)
    if a1[i] != a1[j]:
        rand.append((i, j))
rand = np.array(rand)
base = float(np.mean(cls[rand[:, 0]] == cls[rand[:, 1]]))
print(f"   random cross-subtree same-class rate: {base:.3f}")
print(
    "   cluster  leaves  points  mean profile(5)                 same-class  ratio   top types among same-class pairs"
)
for c in range(KC):
    m = lab == c
    pr = pairs_same_cluster(c)
    if not len(pr):
        continue
    same = cls[pr[:, 0]] == cls[pr[:, 1]]
    rate = float(same.mean())
    tops = np.bincount(cls[pr[same, 0]], minlength=n_types)
    top3 = np.argsort(-tops)[:3]
    prof = np.nanmean(P[m], axis=0)[:5]
    print(
        f"   {c:7d} {m.sum():7d} {size[m].sum() / len(y):6.1%}  {' '.join(f'{v:4.1f}' if np.isfinite(v) else '   .' for v in prof)}   {rate:.3f}     {rate / base:4.2f}x   "
        + ", ".join(f"type {int(t)} ({tops[t] / max(same.sum(), 1):.0%})" for t in top3)
    )

print(
    "\n3. the types that dominate enriched pairs: how split across subtrees, and their family mix (leaf_shape_families)"
)
fam = json.loads((OUT / "leaf_shape_families_results.json").read_text())["brain_1m"]["classes"]
for t in sorted(split, key=lambda t: -split[t]["subtrees"])[:6]:
    mix = fam.get(str(t), {})
    print(
        f"   type {t:3d}: subtrees {split[t]['subtrees']}, largest {split[t]['largest']:.2f}; family mix "
        + (" ".join(f"{k} {v:.0%}" for k, v in mix.items() if v > 0.02) if mix else "n/a")
    )
