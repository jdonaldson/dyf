"""The missing arm: the flat sweep already nests 92-99% of points between consecutive
resolutions (nested_louvain_test.py).  What does it cost to make it EXACTLY nested after the
fact, instead of rebuilding the hierarchy by merging or recursion?

Process coarse -> fine along RES.  Two ways to make level i+1 refine level i (already fixed):
  snap       each fine community moves wholly to the coarse community holding the majority of
             its points (k unchanged, 1-14% of points may move)
  intersect  a fine community that straddles coarse communities is split by coarse membership
             (lossless, k grows by the number of straddlers)
Cost = flat ARI at that level - nested ARI at that level; points moved; k change.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python nested_sweep_snap_test.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def relabel(lab):
    return np.unique(lab, return_inverse=True)[1]


def snap(fine, coarse):
    fine.copy()
    new = np.zeros(fine.max() + 1, dtype=np.int64)
    for f in np.unique(fine):
        new[f] = np.bincount(coarse[fine == f]).argmax()
    # fine communities that share a majority coarse parent stay distinct: key = (parent, f)
    key = new[fine] * (fine.max() + 1) + fine
    return relabel(key), new


def intersect(fine, coarse):
    key = coarse.astype(np.int64) * (fine.max() + 1) + fine
    return relabel(key)


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), n)
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    adj = build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10)
    edges = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
    flat = {}
    for res in RES:
        lab, _ = louvain_communities(L, edges, resolution=res)
        flat[res] = relabel(np.asarray(lab))[item_leaf]
    out = {"levels": []}
    print(f"\n== {name} seed={seed}", flush=True)
    # intersect: top-down, coarse fixed, straddling fine communities split (k grows)
    inter = {RES[0]: flat[RES[0]]}
    for i in range(1, len(RES)):
        inter[RES[i]] = intersect(flat[RES[i]], inter[RES[i - 1]])
    # snap: bottom-up, finest fixed, each coarser level = unions of the (nested) finer level,
    # each finer community going to the flat-coarse community holding most of its points
    snapped = {RES[-1]: flat[RES[-1]]}
    for i in range(len(RES) - 2, -1, -1):
        finer = snapped[RES[i + 1]]
        parent = np.zeros(finer.max() + 1, dtype=np.int64)
        for f in np.unique(finer):
            parent[f] = np.bincount(flat[RES[i]][finer == f]).argmax()
        snapped[RES[i]] = relabel(parent[finer])
    for res in RES:
        fl = flat[res]
        s, it = snapped[res], inter[res]
        row = dict(
            res=res,
            k=int(fl.max() + 1),
            flat=float(ari(y, fl)),
            snap=float(ari(y, s)),
            intersect=float(ari(y, it)),
            k_snap=int(s.max() + 1),
            k_inter=int(it.max() + 1),
            moved=float(1.0 - sum(np.bincount(fl[s == c]).max() for c in np.unique(s)) / n),
        )
        out["levels"].append(row)
        print(
            f"   res {res:<4} k={row['k']:4d} flat {row['flat']:.3f} | snap {row['snap']:.3f} (k {row['k_snap']}, moved {row['moved']:.1%}) "
            f"| intersect {row['intersect']:.3f} (k {row['k_inter']})",
            flush=True,
        )
    out["cost_snap_mean"] = float(np.mean([r["flat"] - r["snap"] for r in out["levels"]]))
    out["cost_intersect_mean"] = float(np.mean([r["flat"] - r["intersect"] for r in out["levels"]]))
    best = max(out["levels"], key=lambda r: r["flat"])
    out["cost_snap_at_best"] = best["flat"] - best["snap"]
    out["cost_intersect_at_best"] = best["flat"] - best["intersect"]
    out["k_inter_growth"] = float(np.mean([r["k_inter"] / r["k"] for r in out["levels"]]))
    print(
        f"   mean cost: snap {out['cost_snap_mean']:+.3f}  intersect {out['cost_intersect_mean']:+.3f}; at best level: "
        f"snap {out['cost_snap_at_best']:+.3f} intersect {out['cost_intersect_at_best']:+.3f}; intersect k growth x{out['k_inter_growth']:.2f}",
        flush=True,
    )
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "nested_sweep_snap_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
