"""Does pairwise shared density detect BRIDGES? The one framing it was not ruled out in.

`sec_shared_density.py` falsified shared density for same-class discrimination and for
transition/boundary detection. It left one framing open, and it is the natural one: a pair
with **dense endpoints and an empty midpoint** is *definitionally* a bridge — two populated
regions joined across a void.

GROUND TRUTH, chosen to be independent of both dyf and of density: **edge betweenness
centrality** on a symmetric kNN graph. That is the standard graph-theoretic definition of a
bridge edge (it counts shortest paths forced through the edge), it is pair-level so it matches
the hypothesis directly, and it knows nothing about local density or about LSH buckets. A high
edge-betweenness edge is a bridge; that is not a proxy, it is the definition.

INCUMBENT, because "does it beat nothing" is the wrong question: dyf already ships
`DensityClassifier.analyze_bridges` (bucket-adjacency, returns `bridge_indices` at the POINT
level) and `find_super_connectors`. Node betweenness is scored against those.

ABLATIONS, in order of how embarrassing it would be to lose to them:
  edge_length     kNN edge length. Long edges span gaps, so this is the confound that could
                  explain everything. Shared density MUST beat it.
  endpoint_min    the lower of the two endpoint densities — per-point density, what dyf
                  gives today.
  mid_over_end    midpoint density / mean endpoint density. THE hypothesis: low = a void
                  between two populated regions.
  segment_min     min density sampled along the edge, for a thin isthmus the midpoint misses.

Scale check first: `sec_shared_density.py` found the measure degenerate when the pair
separation is far below the density radius (mid/endpoint pinned at exactly 1.000). kNN edge
lengths sit near the k-th NN distance, i.e. near the radius, so this regime should be usable —
but the ratio distribution is reported so that is verified rather than assumed.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
N_SUB = 4000
K_NN = 8
K_REF = 10
SEG_STEPS = 5
BT_PIVOTS = 600  # approximate-betweenness pivots
TOP_FRAC = 0.10  # top decile of betweenness = "is a bridge"
SEED = 42


def auc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(float)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def density_at(Q, X, radius, chunk=4096):
    out = np.zeros(len(Q), dtype=np.int64)
    r2 = radius * radius
    qs = (Q * Q).sum(1)
    for c in range(0, len(X), chunk):
        Xc = X[c : c + chunk]
        d2 = qs[:, None] + (Xc * Xc).sum(1)[None, :] - 2.0 * (Q @ Xc.T)
        out += (d2 <= r2).sum(1)
    return out


def analyse(X, name, out):
    import networkx as nx

    print(f"\n{'=' * 78}\n{name}: {X.shape}\n{'=' * 78}", flush=True)

    # pairwise distances on the subsample
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0, out=D2)
    D = np.sqrt(D2)
    np.fill_diagonal(D, np.inf)
    radius = float(np.median(np.partition(D, K_REF, axis=1)[:, K_REF]))

    # symmetric kNN graph
    nn = np.argpartition(D, K_NN, axis=1)[:, :K_NN]
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(X)):
        for j in nn[i]:
            G.add_edge(int(i), int(j), weight=float(D[i, j]))
    print(f"kNN graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, radius={radius:.3f}")

    t0 = time.time()
    eb = nx.edge_betweenness_centrality(G, k=min(BT_PIVOTS, len(X)), seed=SEED)
    nb = nx.betweenness_centrality(G, k=min(BT_PIVOTS, len(X)), seed=SEED)
    print(f"betweenness ({BT_PIVOTS} pivots) in {time.time() - t0:.0f}s", flush=True)

    # ---- edge level: the hypothesis, scored against edge betweenness ----------------
    edges = np.array(list(eb.keys()))
    ebv = np.array([eb[tuple(e)] for e in edges], float)
    A, B = X[edges[:, 0]], X[edges[:, 1]]
    elen = np.linalg.norm(A - B, axis=1)
    dmid = density_at((A + B) / 2.0, X, radius).astype(float)
    da = density_at(A, X, radius).astype(float)
    db = density_at(B, X, radius).astype(float)
    seg = np.stack([density_at(A + (B - A) * t, X, radius) for t in np.linspace(0.2, 0.8, SEG_STEPS)]).astype(float)
    end_mean = (da + db) / 2.0
    ratio = dmid / np.maximum(end_mean, 1e-9)

    print(
        f"\nscale check — mid/endpoint ratio: min {ratio.min():.3f} median {np.median(ratio):.3f} "
        f"max {ratio.max():.3f}; {100 * (ratio == 1.0).mean():.1f}% pinned at exactly 1.0"
    )
    print(
        f"  edge length: median {np.median(elen):.2f} vs density radius {radius:.2f} "
        f"(ratio {np.median(elen) / radius:.2f}) — usable if not << 1"
    )

    is_bridge = ebv >= np.quantile(ebv, 1 - TOP_FRAC)
    feats = {
        "edge_length": elen,
        "-endpoint_min": -np.minimum(da, db),
        "-mid_over_end": -ratio,
        "-segment_min": -seg.min(0),
        "-midpoint": -dmid,
    }
    print(f"\nEDGE level — AUC for 'top {int(100 * TOP_FRAC)}% edge betweenness' ({is_bridge.sum()} of {len(ebv)})")
    print(f"{'feature':<18}{'AUC':>9}{'rho vs eb':>12}")
    rec_e = {}
    for k, v in feats.items():
        a, r = auc(v, is_bridge), spearman(v, ebv)
        rec_e[k] = {"auc": a, "rho": r}
        print(f"{k:<18}{a:>9.3f}{r:>12.3f}")

    # partial: does the ratio survive controlling for edge length?
    def rank_resid(y, x):
        ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
        rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
        M = np.vstack([rx, np.ones_like(rx)]).T
        return ry - M @ np.linalg.lstsq(M, ry, rcond=None)[0]

    pr = spearman(rank_resid(-ratio, elen), rank_resid(ebv, elen))
    print(f"\npartial rho(-mid_over_end, edge_betweenness | edge_length) = {pr:+.3f}")
    print("  ^ this is the number that matters: information beyond edge length")
    rec_e["partial_ratio_given_length"] = pr

    # ---- node level: against dyf's shipped incumbent --------------------------------
    from dyf_rs import DensityClassifier

    clf = DensityClassifier(embedding_dim=X.shape[1], num_bits=10, seed=SEED)
    clf.fit(X)
    ba = clf.analyze_bridges(X)
    dyf_bridge = np.zeros(len(X), bool)
    dyf_bridge[np.asarray(ba.bridge_indices, dtype=np.int64)] = True

    nbv = np.array([nb[i] for i in range(len(X))], float)
    node_is_bridge = nbv >= np.quantile(nbv, 1 - TOP_FRAC)
    own_dens = density_at(X, X, radius).astype(float)
    deg = np.array([G.degree(i) for i in range(len(X))], float)
    # aggregate the edge hypothesis to nodes: a bridge node owns a low-ratio edge
    node_ratio = np.full(len(X), np.inf)
    for (i, j), r_ in zip(edges, ratio):
        node_ratio[i] = min(node_ratio[i], r_)
        node_ratio[j] = min(node_ratio[j], r_)
    node_ratio[~np.isfinite(node_ratio)] = np.median(ratio)

    print(f"\nNODE level — AUC for 'top {int(100 * TOP_FRAC)}% node betweenness'")
    print(f"{'feature':<24}{'AUC':>9}")
    node_feats = {
        "degree": deg,
        "-own_density": -own_dens,
        "dyf analyze_bridges": dyf_bridge.astype(float),
        "-min_edge_ratio": -node_ratio,
    }
    rec_n = {}
    for k, v in node_feats.items():
        a = auc(v, node_is_bridge)
        rec_n[k] = a
        print(f"{k:<24}{a:>9.3f}")
    print(f"  (dyf flags {dyf_bridge.sum()} of {len(X)} points as bridges)")

    out[name] = {"radius": radius, "edge": rec_e, "node": rec_n, "n_edges": int(len(ebv))}


def main():
    rng = np.random.default_rng(SEED)
    out = {}

    E, *_ = S.load()
    idx = rng.choice(len(E), N_SUB, replace=False)
    analyse(np.ascontiguousarray(E[idx]), "sec_768", out)

    p = os.path.join(PAPER, "cmu_mocap_features.npy")
    if os.path.exists(p):
        X = np.load(p).astype(np.float32)
        X = np.ascontiguousarray(X[rng.choice(len(X), N_SUB, replace=False)])
        analyse(X, "cmu_mocap_62", out)

    path = os.path.join(S.CACHE, "bridge_density_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
