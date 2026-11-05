"""Step 1 of the hash-native composition test: build the leaf graph FROM THE HASH, not from
centroid cosine, and see whether Louvain over it (no reassignment) holds up.

Edges come from boundary mass. At every internal node, for every hyperplane bit b, the
points whose |projection| on w_b is in the bottom q% of that node are "near the cut". Each
such point x has its own leaf; flip bit b of x's bucket id, descend the sibling child's
subtree with the stored hyperplanes (exactly as search multiprobe does) to find the leaf x
would land in on the other side; add an edge (leaf(x), leaf'(x)). Symmetrise. Two weightings:
raw counts, and counts / sqrt(n_i n_j).

Arms per (dataset, seed), same tree and same point kNN graph as before:
  cent-block / cent-point   shipped: cosine k=10 leaf-centroid graph, Louvain; then without /
                            with _reassign_points
  hash_raw-block / -point   boundary-mass graph, raw counts
  hash_norm-block / -point  boundary-mass graph, size-normalised
Outcome fixed before running: does hash-block beat cent-block everywhere, and does it hold
MNIST's block ARI while matching cent-point on the single-cell sets?
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python hash_leaf_graph_test.py [seeds...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import louvain_communities
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points, _run_louvain_on_centroids

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import conductance, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
Q = 20.0  # boundary percentile per node per bit
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def index_tree(tree, n):
    """Assign leaf ids; return item_leaf and the list of internal nodes."""
    item_leaf = np.full(n, -1, dtype=np.int64)
    internal = []
    counter = [0]

    def walk(node):
        ch = node.get("children") or []
        if not ch:
            node["leaf_id"] = counter[0]
            item_leaf[np.asarray(node["indices"])] = counter[0]
            counter[0] += 1
            return
        internal.append(node)
        for c in ch:
            walk(c)

    walk(tree)
    return item_leaf, internal


def route(node, Xn, rows):
    """Descend from `node` with rows (global indices) using stored hyperplanes; returns leaf
    id per row. Rows that hit an absent bucket fall back to the child with the nearest
    centroid of the ones present (margin-weighted fallback, as search does)."""
    out = np.full(len(rows), -1, dtype=np.int64)
    if not node.get("children"):
        out[:] = node["leaf_id"]
        return out
    hp = np.asarray(node["hyperplanes"], dtype=np.float32)
    P = Xn[rows] @ hp.T
    bits = (P > 0).astype(np.int64)
    ids = (bits * (1 << np.arange(hp.shape[0]))).sum(1)
    b2c = {int(k): v for k, v in node["bucket_id_to_child"].items()}
    present = np.array(sorted(b2c))
    # fallback: nearest present bucket id in Hamming distance weighted by |margin| (cheap proxy:
    # pick the present bucket whose bits agree with the point on the largest-|P| bits first)
    for bid in np.unique(ids):
        m = ids == bid
        if int(bid) in b2c:
            child = node["children"][b2c[int(bid)]]
            out[m] = route(child, Xn, rows[m])
        else:
            sub = rows[m]
            cost = np.zeros((len(sub), len(present)), dtype=np.float32)
            for j, pb in enumerate(present):
                diff = (bits[m] ^ ((pb >> np.arange(hp.shape[0])) & 1)) == 1
                cost[:, j] = (np.abs(P[m]) * diff).sum(1)
            pick = present[np.argmin(cost, axis=1)]
            for pb in np.unique(pick):
                mm = pick == pb
                out[np.flatnonzero(m)[mm]] = route(node["children"][b2c[int(pb)]], Xn, sub[mm])
    return out


def boundary_edges(internal, item_leaf, Xn, L, q=Q):
    """Accumulate leaf-leaf boundary-mass counts over all internal nodes and bits."""
    M = np.zeros((L, L), dtype=np.float64)
    n_boundary = 0
    for node in internal:
        rows = np.asarray(node["indices"])
        hp = np.asarray(node["hyperplanes"], dtype=np.float32)
        P = Xn[rows] @ hp.T
        bits = (P > 0).astype(np.int64)
        ids = (bits * (1 << np.arange(hp.shape[0]))).sum(1)
        b2c = {int(k): v for k, v in node["bucket_id_to_child"].items()}
        for b in range(hp.shape[0]):
            thr = np.percentile(np.abs(P[:, b]), q)
            near = np.abs(P[:, b]) < thr
            if not near.any():
                continue
            flipped = ids[near] ^ (1 << b)
            src_leaf = item_leaf[rows[near]]
            for fb in np.unique(flipped):
                if int(fb) not in b2c:
                    continue
                m = flipped == fb
                dst_leaf = route(node["children"][b2c[int(fb)]], Xn, rows[near][m])
                np.add.at(M, (src_leaf[m], dst_leaf), 1.0)
                n_boundary += int(m.sum())
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    return M, n_boundary


def louvain_on_matrix(M):
    ii, jj = np.nonzero(M)
    edges = [(int(i), int(j), float(M[i, j])) for i, j in zip(ii, jj)]
    lab, nc = louvain_communities(M.shape[0], edges, resolution=1.0)
    return np.asarray(lab), nc, len(edges) // 2


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    J = np.load(OUT / f"knn15_{name}.npy")
    tree = build_dyf_tree(X, seed=seed, **TREE)
    item_leaf, internal = index_tree(tree, len(X))
    L = item_leaf.max() + 1
    Xn = unit(X)
    leaf_cent = np.zeros((L, X.shape[1]), dtype=np.float32)
    np.add.at(leaf_cent, item_leaf, Xn)
    leaf_cent = unit(leaf_cent)
    sizes = np.bincount(item_leaf, minlength=L).astype(np.float64)
    t1 = time.perf_counter()
    M, n_boundary = boundary_edges(internal, item_leaf, Xn, L)
    t_graph = time.perf_counter() - t1
    graphs = {}
    cl, _, _ = _run_louvain_on_centroids(leaf_cent, k=10, resolution=1.0, similarity_threshold=0.5)
    graphs["cent"] = (np.asarray(cl), None, None)
    lab, nc, ne = louvain_on_matrix(M)
    graphs["hash_raw"] = (lab, nc, ne)
    Mn = M / np.sqrt(np.outer(sizes, sizes))
    lab, nc, ne = louvain_on_matrix(Mn)
    graphs["hash_norm"] = (lab, nc, ne)
    isolated = int((M.sum(1) == 0).sum())
    res = {
        "leaves": int(L),
        "boundary_points": n_boundary,
        "hash_edges": graphs["hash_raw"][2],
        "isolated_leaves": isolated,
        "graph_seconds": t_graph,
    }
    print(
        f"\n== {name} seed={seed}: leaves={L} boundary routings={n_boundary} hash edges={res['hash_edges']} "
        f"isolated leaves={isolated} graph {t_graph:.1f}s",
        flush=True,
    )
    for g, (leaf_lab, _, _) in graphs.items():
        _, leaf_lab = np.unique(leaf_lab, return_inverse=True)
        block = leaf_lab[item_leaf].astype(np.int32)
        point = _reassign_points(block.copy(), X)
        for tag, lab in ((f"{g}-block", block), (f"{g}-point", point)):
            _, lab2 = np.unique(lab, return_inverse=True)
            c = conductance(lab2, J)
            c.update(ari=float(ari(y, lab2)), ami=float(ami(y, lab2)))
            res[tag] = c
            print(
                f"   {tag:15s} k={c['k']:3d} cut={c['cut']:.3f} phi max={c['phi_max']:.3f} "
                f"ARI={c['ari']:.3f} AMI={c['ami']:.3f}",
                flush=True,
            )
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "hash_leaf_graph_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
