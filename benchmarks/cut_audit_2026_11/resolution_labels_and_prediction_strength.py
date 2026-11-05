"""Two ways to pick Louvain's resolution without the full classification.

Shared: library-default tree, shipped cosine k=10 leaf-centroid graph, resolution sweep RES,
block labels per resolution, regret = best-in-sweep ARI (full labels) - chosen ARI.

B  label efficiency: draw n labelled points at random (n in N_LABELS, DRAWS draws), pick the
   resolution whose partition agrees best with them (ARI on the subset), report regret.
   Says what a consumer with a few labels should do.
C  prediction strength (Tibshirani & Walther 2005) through the frozen tree: split points into
   halves A/B; tree + Louvain sweep on A; assign B by A's rule (nearest community centroid,
   cosine -- the shipped reassignment rule); cluster B on its own tree at the same resolution;
   ps(res) = min over B's clusters of the fraction of within-cluster pairs that A's rule also
   puts together.  Selection rules: largest k with ps >= t for t in (0.7, 0.8, 0.9), and
   argmax ps with k > 1.  Label-free; the thresholds are the classical constants and are
   reported side by side rather than chosen.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python resolution_labels_and_prediction_strength.py [seeds]
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
N_LABELS = (10, 20, 50, 100, 200, 500)
DRAWS = 10
PS_T = (0.7, 0.8, 0.9)
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def leaf_graph(Xn, item_leaf):
    L = item_leaf.max() + 1
    cent = np.zeros((L, Xn.shape[1]), dtype=np.float32)
    np.add.at(cent, item_leaf, Xn)
    cent = unit(cent)
    adj = build_knn_graph(np.ascontiguousarray(cent, dtype=np.float32), k=10)
    edges = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
    return edges, L


def sweep(Xn, item_leaf):
    edges, L = leaf_graph(Xn, item_leaf)
    out = {}
    for res in RES:
        lab, nc = louvain_communities(L, edges, resolution=res)
        _, lab = np.unique(np.asarray(lab), return_inverse=True)
        out[res] = (lab[item_leaf].astype(np.int32), int(nc))
    return out


def community_centroids(point_lab, Xn):
    C = point_lab.max() + 1
    cent = np.zeros((C, Xn.shape[1]), dtype=np.float32)
    np.add.at(cent, point_lab, Xn)
    return unit(cent)


def prediction_strength(pred_B, own_B):
    """min over own-clusters of the share of within-cluster pairs co-assigned by pred."""
    ps = []
    for c in np.unique(own_B):
        m = own_B == c
        n = int(m.sum())
        if n < 2:
            continue
        cnt = np.bincount(pred_B[m])
        ps.append(float((cnt * (cnt - 1)).sum() / (n * (n - 1))))
    return min(ps) if ps else 0.0


def run(name, seed):
    t0 = time.perf_counter()
    X, y = load(name)
    n = len(X)
    Xn = unit(X)
    rng = np.random.default_rng(seed)
    full = sweep(Xn, leaf_map(build_dyf_tree(X, seed=seed, **TREE), n))
    full_ari = {res: float(ari(y, lab)) for res, (lab, _) in full.items()}
    best_res = max(RES, key=lambda r: full_ari[r])
    best = full_ari[best_res]
    out = {
        "true_k": int(len(np.unique(y))),
        "sweep": {str(r): dict(k=full[r][1], ari=full_ari[r]) for r in RES},
        "best_res": best_res,
        "best_ari": best,
        "regret_default": best - full_ari[1.0],
    }
    print(
        f"\n== {name} seed={seed}: best res={best_res} (k={full[best_res][1]}) ARI={best:.3f}; "
        f"default regret={out['regret_default']:.3f}",
        flush=True,
    )

    # B: label efficiency
    out["labels"] = {}
    for nl in N_LABELS:
        regrets, picked_k = [], []
        for _ in range(DRAWS):
            idx = rng.choice(n, size=min(nl, n), replace=False)
            pick = max(RES, key=lambda r: ari(y[idx], full[r][0][idx]))
            regrets.append(best - full_ari[pick])
            picked_k.append(full[pick][1])
        out["labels"][str(nl)] = dict(
            regret=float(np.mean(regrets)), regret_sd=float(np.std(regrets)), k=float(np.mean(picked_k))
        )
        print(
            f"   labels n={nl:4d}: regret {np.mean(regrets):.3f} ± {np.std(regrets):.3f} (k {np.mean(picked_k):.0f})",
            flush=True,
        )

    # C: prediction strength through the frozen tree
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2 :]
    sweep_A = sweep(Xn[A], leaf_map(build_dyf_tree(X[A], seed=seed, **TREE), len(A)))
    sweep_B = sweep(Xn[B], leaf_map(build_dyf_tree(X[B], seed=seed + 100, **TREE), len(B)))
    ps = {}
    for res in RES:
        labA, _ = sweep_A[res]
        cent = community_centroids(labA, Xn[A])
        pred_B = np.argmax(Xn[B] @ cent.T, axis=1)
        ps[res] = prediction_strength(pred_B, sweep_B[res][0])
    out["ps"] = {str(r): dict(ps=ps[r], k_A=sweep_A[r][1], k_B=sweep_B[r][1]) for r in RES}
    print("   ps: " + "  ".join(f"r{r}:k{full[r][1]}/{ps[r]:.2f}" for r in RES), flush=True)
    out["ps_pick"] = {}
    for t in PS_T:
        ok = [r for r in RES if ps[r] >= t and full[r][1] > 1]
        pick = max(ok, key=lambda r: full[r][1]) if ok else min(RES)
        out["ps_pick"][f"t{t}"] = dict(res=pick, k=full[pick][1], regret=best - full_ari[pick])
    pick = max((r for r in RES if full[r][1] > 1), key=lambda r: ps[r])
    out["ps_pick"]["argmax"] = dict(res=pick, k=full[pick][1], regret=best - full_ari[pick])
    print(
        "   ps picks: "
        + "  ".join(f"{k}: res {v['res']} k {v['k']} regret {v['regret']:.3f}" for k, v in out["ps_pick"].items()),
        flush=True,
    )
    out["seconds"] = time.perf_counter() - t0
    return out


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [42, 1, 2]
    out_path = OUT / "resolution_labels_and_prediction_strength_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in ORDER:
        for seed in seeds:
            results[f"{name}:{seed}"] = run(name, seed)
            out_path.write_text(json.dumps(results, indent=1))
    print("\nwrote", out_path)
