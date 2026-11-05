"""What does nesting cost?  Three nested constructions vs the flat resolution sweep.

Per (dataset, seed), library-default tree, shipped cosine k=10 leaf-centroid graph:
  flat        Louvain at RES independently -> ARI_flat(k); also 'nestedness' of consecutive
              sweep partitions (point-weighted share of each fine community inside its
              majority coarse community; 1 = perfectly nested)
  merge_cent  bottom-up from the finest sweep partition (res 6): scipy complete linkage on
              unit community centroids (what compute_louvain_hierarchy ships), cut at each
              flat k -> ARI; nesting cost = ARI_flat(k) - ARI_merge(k)
  merge_cut   bottom-up from the same base: greedy merge of the pair with the largest
              normalised cut weight W_cd / sqrt(vol_c vol_d) on the community-level graph
              (leaf-graph edge weights summed), cut at each flat k
  topdown     Louvain at res 1, then Louvain on each community's induced leaf subgraph at
              res r' in (2, 4); compared with flat at the closest k
Per-community persistence (merge_cent): merge step at which a base community is absorbed,
vs that community's label purity (Spearman) -- does persistence predict quality?
Outcome fixed before running: mean nesting cost per construction, and whether the flat sweep
already nests (nestedness).
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python nested_louvain_test.py [seeds]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from dyf_rs import build_knn_graph, louvain_communities
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ari

from dyf import build_dyf_tree

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_conductance_bench import leaf_map, load  # noqa: E402

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RES = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BASE_RESS, TOPDOWN_RES = (3.0, 6.0), (0.7, 1.0, 1.5)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def louvain(edges, L, res):
    lab, nc = louvain_communities(L, edges, resolution=res)
    _, lab = np.unique(np.asarray(lab), return_inverse=True)
    return lab, int(nc)


def nestedness(fine, coarse):
    tot = 0
    for f in np.unique(fine):
        m = fine == f
        tot += np.bincount(coarse[m]).max()
    return tot / len(fine)


def merge_cut_dendrogram(W, vol):
    """Greedy agglomeration on a community graph; returns list of (a, b) merges in order and
    a function to cut at k."""
    C = W.shape[0]
    alive = list(range(C))
    members = {c: [c] for c in range(C)}
    W = W.copy()
    vol = vol.copy().astype(np.float64)
    merges = []
    while len(alive) > 1:
        _best, _pair = -1.0, None
        A = np.array(alive)
        sub = W[np.ix_(A, A)]
        norm = np.sqrt(np.outer(vol[A], vol[A])) + 1e-12
        S = sub / norm
        np.fill_diagonal(S, -1.0)
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        a, b = int(A[i]), int(A[j])
        if S[i, j] <= 0:  # no edges left: merge smallest two
            a, b = sorted(alive, key=lambda c: vol[c])[:2]
        W[a, :] += W[b, :]
        W[:, a] += W[:, b]
        W[a, a] = 0.0
        vol[a] += vol[b]
        members[a] += members[b]
        alive.remove(b)
        merges.append((a, b, list(members[a])))
    return merges


def cut_merges(merges, C, k):
    """Labels per base community after performing merges until k groups remain."""
    lab = np.arange(C)
    n_groups = C
    for a, b, _ in merges:
        if n_groups <= k:
            break
        lab[lab == lab[b]] = lab[a]
        n_groups -= 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab


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

    flat = {res: louvain(edges, L, res) for res in RES}
    flat_pts = {res: flat[res][0][item_leaf] for res in RES}
    flat_ari = {res: float(ari(y, flat_pts[res])) for res in RES}
    nest = [nestedness(flat_pts[RES[i + 1]], flat_pts[RES[i]]) for i in range(len(RES) - 1)]
    out = {
        "leaves": int(L),
        "true_k": int(len(np.unique(y))),
        "flat": {str(r): dict(k=flat[r][1], ari=flat_ari[r]) for r in RES},
        "sweep_nestedness": [float(v) for v in nest],
    }
    print(
        f"\n== {name} seed={seed}: leaves={L}; flat k: {[flat[r][1] for r in RES]}; "
        f"sweep nestedness (consecutive) mean={np.mean(nest):.3f} min={np.min(nest):.3f}",
        flush=True,
    )

    out["bottom_up"], out["nesting_cost"], out["persistence_vs_purity_rho"] = {}, {}, {}
    for base_res in BASE_RESS:
        base_leaf, C = flat[base_res]
        base_pts = base_leaf[item_leaf]
        targets = sorted({flat[r][1] for r in RES if 1 < flat[r][1] < C})
        W = np.zeros((C, C))
        for i, j, w in edges:
            a, b = base_leaf[i], base_leaf[j]
            if a != b:
                W[a, b] += w
                W[b, a] += w
        vol = np.bincount(base_pts, minlength=C).astype(np.float64)
        ccent = unit(np.stack([Xn[base_pts == c].mean(0) for c in range(C)]))
        Z = linkage(ccent, method="complete")
        merges = merge_cut_dendrogram(W, vol)
        costs = {"merge_cent": [], "merge_cut": []}
        rows = []
        print(f"   -- base res {base_res} (k={C})", flush=True)
        for k in targets:
            ref = max(flat_ari[r] for r in RES if flat[r][1] == k)
            lab_c = fcluster(Z, k, criterion="maxclust") - 1
            lab_g = cut_merges(merges, C, k)
            a_c = float(ari(y, lab_c[base_pts]))
            a_g = float(ari(y, lab_g[base_pts]))
            rows.append(dict(k=k, flat=ref, merge_cent=a_c, merge_cut=a_g))
            costs["merge_cent"].append(ref - a_c)
            costs["merge_cut"].append(ref - a_g)
            print(
                f"   k={k:3d}: flat {ref:.3f}  merge_cent {a_c:.3f} ({ref - a_c:+.3f})  merge_cut {a_g:.3f} ({ref - a_g:+.3f})",
                flush=True,
            )
        out["bottom_up"][str(base_res)] = rows
        out["nesting_cost"][str(base_res)] = {m: float(np.mean(v)) if v else None for m, v in costs.items()}
        absorbed = np.zeros(C)
        for step, (a, b) in enumerate(Z[:, :2].astype(int)):
            for cid in (a, b):
                if cid < C:
                    absorbed[cid] = step
        purity = np.array([np.bincount(y[base_pts == c]).max() / max((base_pts == c).sum(), 1) for c in range(C)])
        rho = spearmanr(absorbed, purity).correlation
        out["persistence_vs_purity_rho"][str(base_res)] = float(rho) if np.isfinite(rho) else None
        print(f"   persistence (merge step absorbed) vs purity: Spearman {rho:.2f}", flush=True)

    # top-down
    coarse_leaf, k0 = flat[1.0]
    out["topdown"] = {}
    for r2 in TOPDOWN_RES:
        fine = np.zeros(L, dtype=np.int64)
        nxt = 0
        for c in range(k0):
            nodes = np.flatnonzero(coarse_leaf == c)
            if len(nodes) < 2:
                fine[nodes] = nxt
                nxt += 1
                continue
            remap = {int(v): i for i, v in enumerate(nodes)}
            sub = [(remap[i], remap[j], w) for i, j, w in edges if i in remap and j in remap]
            if not sub:
                fine[nodes] = nxt
                nxt += 1
                continue
            lab, nc = louvain(sub, len(nodes), r2)
            fine[nodes] = lab + nxt
            nxt += nc
        kt = int(fine.max() + 1)
        a_t = float(ari(y, fine[item_leaf]))
        near = min(RES, key=lambda r: abs(flat[r][1] - kt))
        out["topdown"][str(r2)] = dict(
            k=kt, ari=a_t, flat_k=flat[near][1], flat_ari=flat_ari[near], cost=flat_ari[near] - a_t
        )
        print(
            f"   topdown r'={r2}: k={kt} ARI={a_t:.3f} vs flat k={flat[near][1]} ARI={flat_ari[near]:.3f} (cost {flat_ari[near] - a_t:+.3f})",
            flush=True,
        )
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "nested_louvain_test_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
