"""Map equation (Infomap) vs modularity (Louvain) on the same leaf graphs.

The week's open lever is the objective and the community count it returns.  Infomap
minimises the expected description length of a random walk under a two-level code and
picks the number of modules itself.  Per (dataset, seed), library-default tree, two graphs:
  cent     shipped cosine k=10 leaf-centroid kNN (symmetrised by max)
  ceiling  full point kNN coarsened to leaves (counts) -- the reference graph
For each graph:
  louvain@1       shipped objective at resolution 1
  infomap         two-level map equation, undirected weighted, 10 trials
  louvain@k_im    Louvain at the resolution whose k is closest to Infomap's k (separates
                  'better k' from 'better partition at that k')
Scored on block labels (leaf -> module) and after the shipped point reassignment: k, ARI,
AMI vs truth, cut on the point kNN graph.  Also the leaf oracle and true k for context.
Outcome fixed before running: does Infomap's k land nearer the true k than Louvain@1, and at
Infomap's k does it beat Louvain on ARI/AMI?
Env: uv run --no-sync --with infomap --project ~/Projects/dyf python map_equation_test.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from infomap import Infomap
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import conductance, leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def sym_matrix_from_knn(adj, L):
    M = np.zeros((L, L), dtype=np.float64)
    for i, nb in enumerate(adj):
        for j, w in nb:
            M[i, int(j)] = max(M[i, int(j)], float(w))
    return np.maximum(M, M.T)


def louvain_on(M, res):
    ii, jj = np.nonzero(M)
    edges = [(int(i), int(j), float(M[i, j])) for i, j in zip(ii, jj)]
    lab, nc = louvain_communities(M.shape[0], edges, resolution=res)
    _, lab = np.unique(np.asarray(lab), return_inverse=True)
    return lab, int(nc)


def infomap_on(M, seed, trials=10):
    im = Infomap(f"--two-level --silent --num-trials {trials} --seed {seed}")
    ii, jj = np.nonzero(np.triu(M, 1))
    for i, j in zip(ii, jj):
        im.add_link(int(i), int(j), float(M[i, j]))
    im.run()
    lab = np.zeros(M.shape[0], dtype=np.int64)
    seen = np.zeros(M.shape[0], dtype=bool)
    for node in im.nodes:
        lab[node.node_id] = node.module_id
        seen[node.node_id] = True
    lab[~seen] = lab.max() + 1 + np.arange((~seen).sum())  # isolated leaves: own module
    _, lab = np.unique(lab, return_inverse=True)
    return lab, int(lab.max() + 1), float(im.codelength)


def score(leaf_lab, item_leaf, X, y, J):
    block = leaf_lab[item_leaf].astype(np.int32)
    point = _reassign_points(block.copy(), X)
    out = {}
    for tag, lab in (("block", block), ("point", point)):
        _, lab2 = np.unique(lab, return_inverse=True)
        c = conductance(lab2, J)
        out[tag] = dict(k=c["k"], cut=c["cut"], ari=float(ari(y, lab2)), ami=float(ami(y, lab2)))
    return out


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X))
    L = item_leaf.max() + 1
    Xn = unit(X)
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    graphs = {"cent": sym_matrix_from_knn(build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10), L)}
    k = J.shape[1]
    M = np.zeros((L, L), dtype=np.float64)
    np.add.at(M, (np.repeat(item_leaf, k), item_leaf[J.ravel()]), 1.0)
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    graphs["ceiling"] = M
    true_k = int(len(np.unique(y)))
    oracle = np.array([np.bincount(y[item_leaf == l]).argmax() for l in range(L)])[item_leaf]
    res = {"leaves": int(L), "true_k": true_k, "oracle_ari": float(ari(y, oracle))}
    print(f"\n== {name} seed={seed}: leaves={L} true k={true_k} leaf-oracle ARI={res['oracle_ari']:.3f}", flush=True)
    for g, Mg in graphs.items():
        rows = {}
        lab, nc = louvain_on(Mg, 1.0)
        rows["louvain@1"] = {"k": nc, **score(lab, item_leaf, X, y, J)}
        t1 = time.perf_counter()
        lab, nc, cl = infomap_on(Mg, seed)
        rows["infomap"] = {
            "k": nc,
            "codelength": cl,
            "seconds": time.perf_counter() - t1,
            **score(lab, item_leaf, X, y, J),
        }
        sweep = {r: louvain_on(Mg, r) for r in RES}
        best = min(RES, key=lambda r: abs(sweep[r][1] - nc))
        lab, nc2 = sweep[best]
        rows["louvain@k_im"] = {"k": nc2, "res": best, **score(lab, item_leaf, X, y, J)}
        res[g] = rows
        for tag, r in rows.items():
            print(
                f"   {g:8s} {tag:13s} k={r['k']:4d} | block ARI={r['block']['ari']:.3f} AMI={r['block']['ami']:.3f} "
                f"cut={r['block']['cut']:.3f} | point ARI={r['point']['ari']:.3f} AMI={r['point']['ami']:.3f}"
                + (f" | {r['seconds']:.1f}s" if "seconds" in r else ""),
                flush=True,
            )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "map_equation_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
