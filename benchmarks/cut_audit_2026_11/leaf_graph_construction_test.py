"""Which leaf graph should Louvain run on?  Same tree (library defaults), five graphs:

  cent     shipped: cosine k=10 kNN over leaf centroids, cosine weights (dyf_rs build_knn_graph)
  snn      same kNN, edge weight = Jaccard of the two leaves' neighbour sets (shared-neighbour
           graph, the standard pre-Leiden step in scanpy/Seurat); zero new information
  spread   same kNN, weight = exp(-(1 - cos_ij) / (r_i + r_j)), r = 1 - mean cosine of a
           leaf's points to its centroid: leaf-level mutual reachability, uses the cone's width
  ceiling  the full pynndescent k=15 point graph coarsened to leaves (counts of neighbour pairs
           per leaf pair) -- the best any leaf graph can do given these atoms
  sample   the same, but from a random min(20k, n) subset of query rows -- point evidence at
           subsample cost (stands in for running the index's own search on a sample)

Per (dataset, seed): Louvain at resolutions (0.3, 0.5, 1, 1.5, 2, 3) on each graph; the
reported row for each graph is the resolution whose community count is closest to the
shipped arm's at resolution 1 (k-matched).  Block labels (leaf -> community) and the shipped
point reassignment are both scored: ARI / AMI vs truth, cut on the point kNN graph.
Outcome fixed before running: (1) how far is ceiling above cent at matched k; (2) does any
free graph (snn, spread) close part of it; (3) does sample reach ceiling.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python leaf_graph_construction_test.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import conductance, leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.3, 0.5, 1.0, 1.5, 2.0, 3.0)
SAMPLE = 20000
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def louvain(edges, L, res):
    lab, nc = louvain_communities(L, edges, resolution=res)
    _, lab = np.unique(np.asarray(lab), return_inverse=True)
    return lab, int(nc)


def edges_from_matrix(M):
    ii, jj = np.nonzero(M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(ii, jj)]


def build_graphs(Xn, item_leaf, J, L, seed):
    cent = np.zeros((L, Xn.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    sizes = np.bincount(item_leaf, minlength=L)
    cent = unit(cent)
    mean_cos = np.zeros(L)
    np.add.at(mean_cos, item_leaf, (Xn * cent[item_leaf]).sum(1))
    mean_cos /= np.maximum(sizes, 1)
    radius = np.maximum(1.0 - mean_cos, 1e-4)

    adj = build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10)
    nbrs = [set(int(j) for j, _ in nb) for nb in adj]
    graphs = {}
    graphs["cent"] = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
    snn = []
    for i, nb in enumerate(adj):
        for j, _ in nb:
            j = int(j)
            inter = len(nbrs[i] & nbrs[j])
            union = len(nbrs[i] | nbrs[j])
            snn.append((i, j, inter / union if union else 0.0))
    graphs["snn"] = snn
    spread = []
    for i, nb in enumerate(adj):
        for j, w in nb:
            j = int(j)
            spread.append((i, j, float(np.exp(-(1.0 - w) / (radius[i] + radius[j])))))
    graphs["spread"] = spread

    k = J.shape[1]
    src = np.repeat(item_leaf, k)
    dst = item_leaf[J.ravel()]
    M = np.zeros((L, L), dtype=np.float64)
    np.add.at(M, (src, dst), 1.0)
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    graphs["ceiling"] = edges_from_matrix(M)

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(Xn), size=min(SAMPLE, len(Xn)), replace=False)
    Ms = np.zeros((L, L), dtype=np.float64)
    np.add.at(Ms, (np.repeat(item_leaf[rows], k), item_leaf[J[rows].ravel()]), 1.0)
    Ms = Ms + Ms.T
    np.fill_diagonal(Ms, 0.0)
    graphs["sample"] = edges_from_matrix(Ms)
    return graphs, {"isolated_sample": int((Ms.sum(1) == 0).sum()), "isolated_ceiling": int((M.sum(1) == 0).sum())}


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X))
    L = item_leaf.max() + 1
    Xn = unit(X)
    graphs, info = build_graphs(Xn, item_leaf, J, L, seed)
    sweeps = {}
    for g, edges in graphs.items():
        sweeps[g] = {res: louvain(edges, L, res) for res in RES}
    k_ref = sweeps["cent"][1.0][1]
    res_out = {"leaves": int(L), "k_ref": k_ref, **info}
    print(f"\n== {name} seed={seed}: leaves={L} k_ref={k_ref} isolated(sample)={info['isolated_sample']}", flush=True)
    for g in graphs:
        best_res = min(RES, key=lambda r: abs(sweeps[g][r][1] - k_ref))
        leaf_lab, nc = sweeps[g][best_res]
        block = leaf_lab[item_leaf].astype(np.int32)
        point = _reassign_points(block.copy(), X)
        row = {"res": best_res, "k": nc}
        for tag, lab in (("block", block), ("point", point)):
            _, lab2 = np.unique(lab, return_inverse=True)
            c = conductance(lab2, J)
            row[tag] = dict(
                k=c["k"], cut=c["cut"], phi_max=c["phi_max"], ari=float(ari(y, lab2)), ami=float(ami(y, lab2))
            )
        row["sweep"] = {str(r): sweeps[g][r][1] for r in RES}
        res_out[g] = row
        print(
            f"   {g:8s} res={best_res:<4} k={nc:3d} | block ARI={row['block']['ari']:.3f} AMI={row['block']['ami']:.3f} "
            f"cut={row['block']['cut']:.3f} | point ARI={row['point']['ari']:.3f} AMI={row['point']['ami']:.3f}",
            flush=True,
        )
    res_out["seconds"] = time.perf_counter() - t0
    return res_out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "leaf_graph_construction_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
