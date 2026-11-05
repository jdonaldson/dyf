"""Can a label-free criterion pick Louvain's resolution?

Per (dataset, seed), library-default tree, shipped cosine k=10 leaf-centroid graph
(max-symmetrised).  Resolution sweep RES -> Louvain partition, k, block ARI / AMI (labels used
for SCORING only).  Three label-free criteria, each selects one resolution:
  codelength  two-level map-equation description length of the Louvain partition on the graph
              (Infomap with --no-infomap evaluating a given partition); pick the minimum
  stability   drop 10% of edges at random, 5 draws, Louvain at the same resolution; mean pairwise
              ARI between the 5 perturbed partitions; pick the maximum (k > 1 only)
  eigengap    bottom 64 eigenvalues of the symmetric normalised Laplacian; k_gap = argmax of
              lambda_{i+1}/lambda_i for i >= 2 (ratio decoder, tiny eigenvalues floored);
              pick the resolution whose k is closest to k_gap
References: res=1 (shipped default), best-in-sweep (oracle on labels), k-oracle (resolution
whose k is closest to the true k).  Regret = best ARI - chosen ARI.
Env: uv run --no-sync --with infomap --project ~/Projects/dyf python resolution_criterion_test.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from infomap import Infomap
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DRAWS, DROP, NEIG = 5, 0.10, 64
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def sym_from_knn(adj, L):
    M = np.zeros((L, L), dtype=np.float64)
    for i, nb in enumerate(adj):
        for j, w in nb:
            M[i, int(j)] = max(M[i, int(j)], float(w))
    return np.maximum(M, M.T)


def edges_of(M):
    ii, jj = np.nonzero(M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(ii, jj)]


def louvain_edges(edges, L, res):
    lab, nc = louvain_communities(L, edges, resolution=res)
    _, lab = np.unique(np.asarray(lab), return_inverse=True)
    return lab, int(nc)


def codelength_of(M, lab):
    im = Infomap("--two-level --silent --no-infomap")
    ii, jj = np.nonzero(np.triu(M, 1))
    for i, j in zip(ii, jj):
        im.add_link(int(i), int(j), float(M[i, j]))
    im.run(initial_partition={int(i): int(lab[i]) + 1 for i in range(len(lab))})
    return float(im.codelength)


def stability_of(M, L, res, rng):
    ii, jj = np.nonzero(np.triu(M, 1))
    parts = []
    for _ in range(DRAWS):
        keep = rng.random(len(ii)) >= DROP
        e = [(int(i), int(j), float(M[i, j])) for i, j in zip(ii[keep], jj[keep])]
        e += [(j, i, w) for i, j, w in e]
        parts.append(louvain_edges(e, L, res)[0])
    vals = [ari(parts[a], parts[b]) for a in range(DRAWS) for b in range(a + 1, DRAWS)]
    return float(np.mean(vals))


def eigengap_k(M):
    A = csr_matrix(M)
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    Dm = diags(1.0 / np.sqrt(d))
    Lap = diags(np.ones(len(d))) - Dm @ A @ Dm
    k = min(NEIG, len(d) - 2)
    vals = np.sort(eigsh(Lap, k=k, which="SA", tol=1e-5)[0])
    floor = max(1e-3 * vals[-1], 1e-9)
    v = np.maximum(vals, floor)
    ratios = v[1:] / v[:-1]
    i = int(np.argmax(ratios[1:]) + 1)  # i >= 2 in 1-based terms: ratio lambda_{i+1}/lambda_i
    return i + 1, vals[:8].tolist()


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    true_k = int(len(np.unique(y)))
    item_leaf = leaf_map(build_dyf_tree(X, seed=seed, **TREE), len(X))
    L = item_leaf.max() + 1
    Xn = unit(X)
    cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    M = sym_from_knn(build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10), L)
    edges = edges_of(M)
    rng = np.random.default_rng(seed)
    sweep = []
    for res in RES:
        lab, nc = louvain_edges(edges, L, res)
        block = lab[item_leaf]
        sweep.append(
            dict(
                res=res,
                k=nc,
                ari=float(ari(y, block)),
                ami=float(ami(y, block)),
                codelength=codelength_of(M, lab),
                stability=stability_of(M, L, res, rng),
            )
        )
    k_gap, eig = eigengap_k(M)
    pick = {
        "default": next(s for s in sweep if s["res"] == 1.0),
        "best": max(sweep, key=lambda s: s["ari"]),
        "k_oracle": min(sweep, key=lambda s: abs(s["k"] - true_k)),
        "codelength": min(sweep, key=lambda s: s["codelength"]),
        "stability": max((s for s in sweep if s["k"] > 1), key=lambda s: s["stability"]),
        "eigengap": min(sweep, key=lambda s: abs(s["k"] - k_gap)),
    }
    out = {
        "leaves": int(L),
        "true_k": true_k,
        "k_gap": int(k_gap),
        "eig": eig,
        "sweep": sweep,
        "pick": {
            c: dict(res=s["res"], k=s["k"], ari=s["ari"], ami=s["ami"], regret=pick["best"]["ari"] - s["ari"])
            for c, s in pick.items()
        },
    }
    print(f"\n== {name} seed={seed}: leaves={L} true k={true_k} eigengap k={k_gap}", flush=True)
    print(
        "   sweep: "
        + "  ".join(
            f"r{s['res']}:k{s['k']}/{s['ari']:.2f}/cl{s['codelength']:.2f}/st{s['stability']:.2f}" for s in sweep
        ),
        flush=True,
    )
    for c, s in out["pick"].items():
        print(
            f"   {c:10s} res={s['res']:<4} k={s['k']:4d} ARI={s['ari']:.3f} AMI={s['ami']:.3f} regret={s['regret']:.3f}",
            flush=True,
        )
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "resolution_criterion_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
