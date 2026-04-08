"""
IVF Probe Routing: Bridge-Guided Cluster Search

Standard IVF (Inverted File) indexes assign each point to its nearest cluster
centroid. At query time, you search nprobe clusters — typically the ones whose
centroids are closest to the query vector.

But centroid distance is a coarse signal. Two clusters might have close
centroids without sharing a real boundary. Multi-address bridge points sit
on actual cluster boundaries and can serve as local navigation beacons.

This script tests query-local bridge routing: for each query, find the
nearest bridge points and probe the clusters they belong to. If the query
sits near a cluster boundary, the nearest bridges will point to the right
secondary clusters — potentially better than centroid distance alone.

Probe strategies:
  1. Centroid (std IVF): nprobe nearest clusters by centroid distance
  2. Bridge-local: find nearest bridge points, probe their clusters
  3. Hybrid: blend centroid rank and bridge-cluster vote
  4. Oracle: cheat — pick clusters containing the most true neighbors

Usage:
    python demo/ivf_probe_routing.py demo/wiki_simple_50k.parquet --sample 10000
"""

import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from dyf import cut_tree_to_labels
from dyf.pca_tree import build_pca_tree, extract_boundary_persistence


def ivf_search(query_emb, cluster_point_lists, embeddings_normed,
               probe_clusters, top_k=10):
    """Search specific clusters, return top-k by cosine similarity.

    Returns (result_indices, n_distance_computations).
    """
    candidates = []
    n_dist = 0

    for c in probe_clusters:
        pts = cluster_point_lists.get(c, [])
        if len(pts) == 0:
            continue
        pts_arr = np.array(pts)
        sims = embeddings_normed[pts_arr] @ query_emb
        n_dist += len(pts)
        for idx_in_list, sim in enumerate(sims):
            candidates.append((float(sim), int(pts_arr[idx_in_list])))

    candidates.sort(reverse=True)
    result_indices = [idx for _, idx in candidates[:top_k]]
    return result_indices, n_dist


def print_table(title, headers, rows):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(
        description="Compare IVF probe routing: centroid vs bridge-guided")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="Number of IVF clusters")
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="Number of query points to evaluate")
    parser.add_argument("--knn-k", type=int, default=20,
                        help="kNN k for ground truth computation")
    parser.add_argument("--bridge-k", type=int, default=30,
                        help="How many nearest bridges to consider per query")
    args = parser.parse_args()

    # ── Load & dedup ─────────────────────────────────────────────────────
    print(f"Loading {args.parquet_path}...")
    df = pl.read_parquet(args.parquet_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, seed=42)

    titles_all = df["title"].to_list()
    embeddings_all = np.array(df["embedding"].to_list(), dtype=np.float32)

    from dyf.chunks import deduplicate_chunks
    from dyf_rs import DensityClassifier as RustClassifier

    clf = RustClassifier(
        embedding_dim=embeddings_all.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings_all)
    bucket_ids = clf.get_bucket_ids()
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles_all))

    embeddings = embeddings_all[dedup_mask]
    n = len(embeddings)
    print(f"  {len(titles_all)} -> {n} after dedup")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # ── Multi-address bridge scores ──────────────────────────────────────
    print(f"\nBuilding PCA tree (depth={args.max_depth})...")
    t0 = time.time()
    tree = build_pca_tree(embeddings, args.max_depth)
    print(f"  Built in {time.time() - t0:.1f}s")

    ma_result = extract_boundary_persistence(tree, margin_pct=args.margin_pct)
    boundary_depths = ma_result['boundary_depths']

    ma_score = np.zeros(n, dtype=np.float64)
    for pt_idx, depths in boundary_depths.items():
        ma_score[pt_idx] = sum(args.max_depth - d for d in depths)

    # Select bridge points: score > 0 means boundary at some depth
    bridge_mask = ma_score > 0
    bridge_indices = np.where(bridge_mask)[0]
    n_bridges = len(bridge_indices)
    print(f"  Bridge points: {n_bridges}/{n} ({100*n_bridges/n:.1f}%)")
    print(f"  Score range: median={np.median(ma_score[bridge_mask]):.1f}, "
          f"max={ma_score.max():.1f}")

    # Also select "strong" bridges — top quartile by score
    score_threshold = np.percentile(ma_score[bridge_mask], 75)
    strong_bridge_mask = ma_score >= score_threshold
    strong_bridge_indices = np.where(strong_bridge_mask)[0]
    n_strong = len(strong_bridge_indices)
    print(f"  Strong bridges (score >= {score_threshold:.0f}): "
          f"{n_strong}/{n} ({100*n_strong/n:.1f}%)")

    # ── Cluster assignment ───────────────────────────────────────────────
    print(f"\nCutting PCA tree to {args.n_clusters} clusters...")
    cluster_labels = cut_tree_to_labels(
        tree, n, args.n_clusters, max_depth=args.max_depth)
    n_actual = len(set(cluster_labels.tolist()))
    print(f"  Got {n_actual} clusters")

    # Build per-cluster point lists and centroids
    cluster_point_lists = defaultdict(list)
    for i in range(n):
        cluster_point_lists[cluster_labels[i]].append(i)

    # Ensure contiguous cluster IDs
    cluster_ids = sorted(cluster_point_lists.keys())
    cluster_centroids = np.zeros((len(cluster_ids), embeddings.shape[1]),
                                  dtype=np.float64)
    for ci, cid in enumerate(cluster_ids):
        pts = cluster_point_lists[cid]
        cluster_centroids[ci] = emb_normed[pts].mean(axis=0)
    centroid_norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
    cluster_centroids_normed = (cluster_centroids /
                                 np.maximum(centroid_norms, 1e-10))

    # Remap cluster labels to contiguous 0..n_actual-1
    id_map = {cid: ci for ci, cid in enumerate(cluster_ids)}
    cluster_labels_mapped = np.array(
        [id_map[cluster_labels[i]] for i in range(n)], dtype=int)
    cluster_point_lists_mapped = {
        ci: cluster_point_lists[cid] for ci, cid in enumerate(cluster_ids)}

    sizes = [len(cluster_point_lists_mapped[c]) for c in range(n_actual)]
    print(f"  Cluster sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
          f"max={max(sizes)}")

    # ── Build bridge kNN index ───────────────────────────────────────────
    # For query-local routing: quickly find nearest bridge points
    print(f"\nBuilding bridge-point kNN index ({n_strong} strong bridges)...")
    bridge_embs = emb_normed[strong_bridge_indices]
    bridge_clusters = cluster_labels_mapped[strong_bridge_indices]

    bridge_nn = NearestNeighbors(
        n_neighbors=min(args.bridge_k, n_strong), metric='cosine')
    bridge_nn.fit(embeddings[strong_bridge_indices])  # use unnormed for cosine

    # ── Ground truth kNN ─────────────────────────────────────────────────
    print(f"\nComputing brute-force k={args.knn_k} nearest neighbors...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=args.knn_k + 1, metric='cosine')
    nn.fit(embeddings)
    _, knn_indices = nn.kneighbors(embeddings)
    knn_indices = knn_indices[:, 1:]  # drop self
    print(f"  Done in {time.time() - t0:.1f}s")

    true_knn = knn_indices[:, :10]

    # ── Select query points ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_queries = min(args.n_queries, n)
    query_indices = rng.choice(n, n_queries, replace=False)

    # Oracle cluster rankings per query
    query_cluster_recall = np.zeros((n_queries, n_actual), dtype=np.float64)
    for qi, q in enumerate(query_indices):
        true_set = set(true_knn[q].tolist())
        for c in range(n_actual):
            pts_in_c = set(cluster_point_lists_mapped[c])
            query_cluster_recall[qi, c] = len(true_set & pts_in_c)

    print(f"\nEvaluating {n_queries} queries...")

    # ── Define probe strategies ──────────────────────────────────────────
    nprobe_values = [1, 2, 3, 4, 6, 8, 12, 16]

    def probe_centroid(q_emb, nprobe):
        """Standard IVF: nearest centroids."""
        sims = cluster_centroids_normed @ q_emb
        return np.argsort(-sims)[:nprobe].tolist()

    def probe_bridge_local(q_emb, q_raw, nprobe):
        """Find nearest bridge points, vote on which clusters to probe."""
        # Primary cluster is always the nearest centroid
        sims = cluster_centroids_normed @ q_emb
        primary = int(np.argmax(sims))

        if nprobe == 1:
            return [primary]

        # Find nearest bridge points
        _, br_idx = bridge_nn.kneighbors(q_raw.reshape(1, -1))
        br_idx = br_idx[0]

        # Vote: which clusters do the nearest bridges belong to?
        cluster_votes = Counter()
        for bi in br_idx:
            c = int(bridge_clusters[bi])
            if c != primary:
                cluster_votes[c] += 1

        # Return primary + top-voted secondary clusters
        secondary = [c for c, _ in cluster_votes.most_common(nprobe - 1)]
        return [primary] + secondary

    def probe_hybrid(q_emb, q_raw, nprobe):
        """Blend centroid rank and bridge votes."""
        # Centroid similarities → ranks
        sims = cluster_centroids_normed @ q_emb
        centroid_order = np.argsort(-sims)
        centroid_rank = np.empty(n_actual, dtype=float)
        for rank, c in enumerate(centroid_order):
            centroid_rank[c] = rank

        primary = int(centroid_order[0])

        if nprobe == 1:
            return [primary]

        # Bridge votes
        _, br_idx = bridge_nn.kneighbors(q_raw.reshape(1, -1))
        br_idx = br_idx[0]

        cluster_votes = Counter()
        for bi in br_idx:
            c = int(bridge_clusters[bi])
            cluster_votes[c] += 1

        max_votes = max(cluster_votes.values()) if cluster_votes else 1

        # Combined score: lower is better
        # Centroid rank (0 = best) + penalty for no bridge votes
        combined = np.zeros(n_actual)
        for c in range(n_actual):
            votes = cluster_votes.get(c, 0)
            # Normalize: centroid_rank in [0, n_actual), votes in [0, max_votes]
            combined[c] = centroid_rank[c] - (votes / max_votes) * n_actual * 0.3

        combined[primary] = -999  # always include primary
        return np.argsort(combined)[:nprobe].tolist()

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = {name: {} for name in
               ["Centroid", "Bridge-local", "Hybrid", "Oracle"]}

    for nprobe in nprobe_values:
        print(f"\n  nprobe={nprobe}:")

        for name in results:
            recalls = []
            dist_counts = []

            for qi, q in enumerate(query_indices):
                true_set = set(true_knn[q].tolist())
                q_emb = emb_normed[q]
                q_raw = embeddings[q]

                if name == "Oracle":
                    probe = np.argsort(
                        -query_cluster_recall[qi])[:nprobe].tolist()
                elif name == "Centroid":
                    probe = probe_centroid(q_emb, nprobe)
                elif name == "Bridge-local":
                    probe = probe_bridge_local(q_emb, q_raw, nprobe)
                elif name == "Hybrid":
                    probe = probe_hybrid(q_emb, q_raw, nprobe)

                found, n_dist = ivf_search(
                    q_emb, cluster_point_lists_mapped,
                    emb_normed, probe, top_k=10)

                found_set = set(found) - {q}
                recall = len(true_set & found_set) / 10.0
                recalls.append(recall)
                dist_counts.append(n_dist)

            mean_recall = np.mean(recalls)
            mean_dist = np.mean(dist_counts)
            results[name][nprobe] = (mean_recall, mean_dist)
            print(f"    {name:15s}: recall@10={mean_recall:.3f}  "
                  f"avg_dist={mean_dist:.0f}")

    # ── Summary tables ───────────────────────────────────────────────────
    recall_rows = []
    for nprobe in nprobe_values:
        row = [str(nprobe)]
        for name in results:
            r, d = results[name][nprobe]
            row.append(f"{r:.3f}")
        recall_rows.append(row)

    print_table(
        "Recall@10 by nprobe",
        ["nprobe"] + list(results.keys()),
        recall_rows,
    )

    # Recall at matched compute: bridge routing may touch different-sized
    # clusters, so normalize by distance computations
    eff_rows = []
    for nprobe in nprobe_values:
        row = [str(nprobe)]
        for name in results:
            r, d = results[name][nprobe]
            eff = r / (d / 1000) if d > 0 else 0
            row.append(f"{eff:.2f}")
        eff_rows.append(row)

    print_table(
        "Efficiency: Recall@10 per 1000 distance computations",
        ["nprobe"] + list(results.keys()),
        eff_rows,
    )

    # ── Per-query analysis at nprobe=4 ───────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Per-query analysis (nprobe=4)")
    print(f"{'=' * 78}")

    nprobe_detail = 4
    centroid_recalls = []
    bridge_recalls = []
    hybrid_recalls = []
    query_ma_scores = []

    for qi, q in enumerate(query_indices):
        true_set = set(true_knn[q].tolist())
        q_emb = emb_normed[q]
        q_raw = embeddings[q]

        probe_c = probe_centroid(q_emb, nprobe_detail)
        found_c, _ = ivf_search(q_emb, cluster_point_lists_mapped,
                                 emb_normed, probe_c, top_k=10)
        r_c = len((set(found_c) - {q}) & true_set) / 10.0

        probe_b = probe_bridge_local(q_emb, q_raw, nprobe_detail)
        found_b, _ = ivf_search(q_emb, cluster_point_lists_mapped,
                                 emb_normed, probe_b, top_k=10)
        r_b = len((set(found_b) - {q}) & true_set) / 10.0

        probe_h = probe_hybrid(q_emb, q_raw, nprobe_detail)
        found_h, _ = ivf_search(q_emb, cluster_point_lists_mapped,
                                 emb_normed, probe_h, top_k=10)
        r_h = len((set(found_h) - {q}) & true_set) / 10.0

        centroid_recalls.append(r_c)
        bridge_recalls.append(r_b)
        hybrid_recalls.append(r_h)
        query_ma_scores.append(ma_score[q])

    centroid_recalls = np.array(centroid_recalls)
    bridge_recalls = np.array(bridge_recalls)
    hybrid_recalls = np.array(hybrid_recalls)
    query_ma_scores = np.array(query_ma_scores)

    for cmp_name, cmp_recalls in [("Bridge-local", bridge_recalls),
                                   ("Hybrid", hybrid_recalls)]:
        wins = np.sum(cmp_recalls > centroid_recalls)
        losses = np.sum(centroid_recalls > cmp_recalls)
        ties = np.sum(cmp_recalls == centroid_recalls)
        print(f"\n  {cmp_name} vs Centroid: "
              f"wins={wins} losses={losses} ties={ties}")

    # Stratify by query's cross-cluster fraction
    query_cross_frac = np.zeros(n_queries)
    for qi, q in enumerate(query_indices):
        q_cluster = cluster_labels_mapped[q]
        nb_clusters = cluster_labels_mapped[true_knn[q]]
        query_cross_frac[qi] = np.mean(nb_clusters != q_cluster)

    high_cross = query_cross_frac > np.percentile(query_cross_frac, 75)
    low_cross = query_cross_frac < np.percentile(query_cross_frac, 25)
    print(f"\n  Boundary queries (top 25% cross-cluster, n={high_cross.sum()}):")
    print(f"    Centroid:     {centroid_recalls[high_cross].mean():.3f}")
    print(f"    Bridge-local: {bridge_recalls[high_cross].mean():.3f}")
    print(f"    Hybrid:       {hybrid_recalls[high_cross].mean():.3f}")
    print(f"  Core queries (bottom 25% cross-cluster, n={low_cross.sum()}):")
    print(f"    Centroid:     {centroid_recalls[low_cross].mean():.3f}")
    print(f"    Bridge-local: {bridge_recalls[low_cross].mean():.3f}")
    print(f"    Hybrid:       {hybrid_recalls[low_cross].mean():.3f}")

    print()


if __name__ == "__main__":
    main()
