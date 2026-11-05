"""What does the final reassignment step do to the partition's cut?

louvain_cluster_leaves ends with _reassign_points: up to 5 rounds of spherical k-means over
ALL points, seeded by the Louvain community labels.  So the shipped partition is a Voronoi
tessellation of community centroids.  Measure, per dataset, on the cached kNN graph:
  block   = Louvain leaf labels mapped to points (before reassignment)
  final   = after _reassign_points
  changed = fraction of points whose label moved
  cut / phi for each, and ARI/AMI vs ground truth for each.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import conductance, leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def run(name, seed=42):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X))
    L = item_leaf.max() + 1
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    cent = np.stack([Xn[item_leaf == l].mean(0) for l in range(L)])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
    leaf_lab, _, _ = _run_louvain_on_centroids(cent.astype(np.float32), k=10, resolution=1.0, similarity_threshold=0.5)
    block = np.asarray(leaf_lab)[item_leaf].astype(np.int32)
    final = _reassign_points(block.copy(), X)
    res = {"changed": float((block != final).mean())}
    print(f"\n== {name} seed={seed}: n={len(X)} leaves={L}  reassignment moved {res['changed']:.1%} of points")
    for tag, lab in (("block", block), ("final", final)):
        c = conductance(lab, J)
        c.update(ari=float(ari(y, lab)), ami=float(ami(y, lab)))
        res[tag] = c
        print(
            f"   {tag:6s} k={c['k']:3d} cut={c['cut']:.3f} phi med={c['phi_median']:.3f} max={c['phi_max']:.3f} "
            f"ARI={c['ari']:.3f} AMI={c['ami']:.3f}",
            flush=True,
        )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "reassignment_effect_seeds_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
