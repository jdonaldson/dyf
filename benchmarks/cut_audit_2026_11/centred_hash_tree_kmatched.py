"""k-matched control for centred_hash_tree_test.py: sweep Louvain resolution on both trees
(py-uncentred, py-centred) and compare block / point ARI at the closest community count.
Reports, per dataset, the best-matching pair for each uncentred k, and the mean ARI delta
(centred - uncentred) over matched pairs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from centred_hash_tree_test import ORDER, TREE, py_tree_leaves, unit  # noqa: E402
from community_conductance_bench import load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
RES = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)


def sweep(X, Xn, item_leaf, y):
    L = item_leaf.max() + 1
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    rows = []
    for res in RES:
        ll, nc, _ = _run_louvain_on_centroids(cent, k=10, resolution=res, similarity_threshold=0.5)
        _, ll = np.unique(np.asarray(ll), return_inverse=True)
        block = ll[item_leaf].astype(np.int32)
        point = _reassign_points(block.copy(), X)
        rows.append(
            dict(
                res=res,
                k=int(nc),
                block_ari=float(ari(y, block)),
                block_ami=float(ami(y, block)),
                point_ari=float(ari(y, point)),
                point_ami=float(ami(y, point)),
                point_k=int(len(np.unique(point))),
            )
        )
    return rows


def match(u_rows, c_rows):
    """For each uncentred row, the centred row with the closest k; returns deltas."""
    out = []
    for u in u_rows:
        c = min(c_rows, key=lambda r: abs(r["k"] - u["k"]))
        if abs(c["k"] - u["k"]) <= max(2, 0.15 * u["k"]):
            out.append(
                dict(
                    k_u=u["k"],
                    k_c=c["k"],
                    d_block=c["block_ari"] - u["block_ari"],
                    d_point=c["point_ari"] - u["point_ari"],
                    u_block=u["block_ari"],
                    c_block=c["block_ari"],
                    u_point=u["point_ari"],
                    c_point=c["point_ari"],
                )
            )
    return out


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    Xn = unit(X)
    res = {}
    for centred in (False, True):
        tag = "centred" if centred else "uncentred"
        res[tag] = sweep(X, Xn, py_tree_leaves(Xn, centred, seed=seed, **TREE), y)
    pairs = match(res["uncentred"], res["centred"])
    res["pairs"] = pairs
    print(f"\n== {name} seed={seed}: {len(pairs)} k-matched pairs", flush=True)
    for p in pairs:
        print(
            f"   k={p['k_u']:3d}/{p['k_c']:3d}  block {p['u_block']:.3f}->{p['c_block']:.3f} ({p['d_block']:+.3f})  "
            f"point {p['u_point']:.3f}->{p['c_point']:.3f} ({p['d_point']:+.3f})",
            flush=True,
        )
    if pairs:
        print(
            f"   mean delta: block {np.mean([p['d_block'] for p in pairs]):+.3f}  "
            f"point {np.mean([p['d_point'] for p in pairs]):+.3f}",
            flush=True,
        )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "centred_hash_tree_kmatched_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
