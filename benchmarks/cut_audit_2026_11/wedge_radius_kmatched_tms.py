"""k-matched control for the band gain on TMS: is the ARI rise from the norm bands,
or just from Louvain returning more communities?  Sweep resolution on the plain-leaf
cosine graph until k matches the band arm, and compare at equal k.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from wedge_radius_bench import (
    OUT,
    TREE,
    K,
    ami,
    ari,
    build_dyf_tree,
    build_knn_graph,
    centroids,
    leaf_map,
    load,
    louvain_communities,
    refine,
)

res = {}
for name in ["tms_facs", "tms_droplet"]:
    X, y, _ = load(name)
    item_leaf = leaf_map(build_dyf_tree(X, seed=42, **TREE), len(X))
    sub, *_ = refine(X, item_leaf)
    rows = []
    for label, groups in (("leaves", item_leaf), ("bands", sub)):
        C, remap = centroids(X, groups)
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        adj = build_knn_graph(np.ascontiguousarray(Cn, dtype=np.float32), k=K)
        edges = [(i, int(j), float(w)) for i, nb in enumerate(adj) for j, w in nb]
        for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            lab, nc = louvain_communities(len(C), edges, resolution=r)
            pl = np.asarray(lab)[remap]
            rows.append(dict(arm=label, res=r, k=int(nc), ari=float(ari(y, pl)), ami=float(ami(y, pl))))
            print(
                f"{name:12s} {label:6s} res={r:.1f} k={nc:4d} ARI={rows[-1]['ari']:.3f} AMI={rows[-1]['ami']:.3f}",
                flush=True,
            )
    res[name] = rows
(OUT / "wedge_radius_kmatched_tms_results.json").write_text(json.dumps(res, indent=1))
