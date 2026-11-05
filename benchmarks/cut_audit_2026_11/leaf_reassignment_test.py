"""Does leaf-level reassignment keep the straggler repair without flattening Louvain?

Arms per (dataset, seed), all from the same tree + Louvain block labels:
  block        Louvain leaf labels mapped to points (no reassignment)
  point        shipped: _reassign_points, up to 5 Lloyd rounds of spherical k-means over points
  leaf         move a WHOLE leaf to the community whose centroid its own centroid is most
               similar to; recompute community centroids; up to 5 rounds
  leaf+point   leaf arm, then ONE point-level round (stragglers inside leaves)
Metrics on the cached pynndescent k=15 graph: cut, worst-community phi, ARI/AMI vs truth,
fraction of points moved relative to block.  Outcome variable fixed before running: per
dataset, does `leaf` keep >= the shipped arm's ARI where shipped helps (TMS, brain) AND keep
the block arm's ARI where shipped hurts (MNIST)?  Three seeds; report min-max.
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


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def community_centroids(labels_per_point, Xn, C):
    cent = np.zeros((C, Xn.shape[1]), dtype=np.float32)
    np.add.at(cent, labels_per_point, Xn)
    return unit(cent)


def reassign_leaves(leaf_lab, leaf_cent, item_leaf, Xn, rounds=5):
    """leaf_lab: community per leaf. Returns new leaf_lab and rounds used."""
    leaf_lab = leaf_lab.copy()
    C = leaf_lab.max() + 1
    for r in range(rounds):
        point_lab = leaf_lab[item_leaf]
        cent = community_centroids(point_lab, Xn, C)
        new = np.argmax(leaf_cent @ cent.T, axis=1)
        changed = int((new != leaf_lab).sum())
        leaf_lab = new
        if changed == 0:
            break
    return leaf_lab, r + 1


def reassign_stragglers(leaf_lab, leaf_cent, rounds=5):
    """Move only leaves with NO same-community neighbour in the k=10 leaf kNN graph (the
    graph Louvain ran on) to the majority community among their neighbours."""
    from dyf_rs import build_knn_graph

    adj = build_knn_graph(np.ascontiguousarray(leaf_cent, dtype=np.float32), k=10)
    nbrs = [np.array([int(j) for j, _ in nb], dtype=np.int64) for nb in adj]
    leaf_lab = leaf_lab.copy()
    total = 0
    for r in range(rounds):
        moved = 0
        new = leaf_lab.copy()
        for i, nb in enumerate(nbrs):
            if len(nb) == 0:
                continue
            nl = leaf_lab[nb]
            if (nl == leaf_lab[i]).any():
                continue
            new[i] = np.bincount(nl).argmax()
            moved += 1
        leaf_lab = new
        total += moved
        if moved == 0:
            break
    return leaf_lab, total


def one_point_round(point_lab, Xn):
    C = point_lab.max() + 1
    cent = community_centroids(point_lab, Xn, C)
    return np.argmax(Xn @ cent.T, axis=1).astype(np.int32)


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X))
    L = item_leaf.max() + 1
    Xn = unit(X)
    leaf_cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(leaf_cent, item_leaf, Xn)
    leaf_cent = unit(leaf_cent)
    leaf_lab, _, _ = _run_louvain_on_centroids(
        leaf_cent.astype(np.float32), k=10, resolution=1.0, similarity_threshold=0.5
    )
    leaf_lab = np.asarray(leaf_lab)
    _, leaf_lab = np.unique(leaf_lab, return_inverse=True)
    block = leaf_lab[item_leaf].astype(np.int32)
    arms = {"block": block}
    arms["point"] = _reassign_points(block.copy(), X)
    new_leaf_lab, rounds = reassign_leaves(leaf_lab, leaf_cent, item_leaf, Xn)
    arms["leaf"] = new_leaf_lab[item_leaf].astype(np.int32)
    arms["leaf+point"] = one_point_round(arms["leaf"], Xn)
    strag_lab, n_strag = reassign_stragglers(leaf_lab, leaf_cent)
    arms["straggler"] = strag_lab[item_leaf].astype(np.int32)
    res = {
        "leaf_rounds": rounds,
        "leaves_moved": float((new_leaf_lab != leaf_lab).mean()),
        "straggler_leaves": int(n_strag),
        "straggler_frac": float(n_strag / L),
    }
    print(
        f"\n== {name} seed={seed}: leaves={L}, leaf arm moved {res['leaves_moved']:.1%} of leaves in {rounds} rounds; "
        f"straggler arm moved {n_strag} leaves ({n_strag / L:.1%})"
    )
    for tag, lab in arms.items():
        moved = float((lab != block).mean())
        _, lab = np.unique(lab, return_inverse=True)
        c = conductance(lab, J)
        c.update(ari=float(ari(y, lab)), ami=float(ami(y, lab)), moved=moved)
        res[tag] = c
        print(
            f"   {tag:10s} k={c['k']:3d} moved={c['moved']:5.1%} cut={c['cut']:.3f} phi max={c['phi_max']:.3f} "
            f"ARI={c['ari']:.3f} AMI={c['ami']:.3f}",
            flush=True,
        )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "leaf_reassignment_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
