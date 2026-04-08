"""
Bridge Comparison: Simple LSH vs Multi-Address PCA Tree

In any dataset, each point has natural nearest neighbors — the records most
similar to it in meaning. When we group records into clusters, about 60% of a
typical point's closest neighbors end up in a different cluster. That's normal —
clusters have edges, and neighbors spill across them.

DYF's LSH hashing can identify points that are far from their bucket center —
potential bridges. These bump the cross-cluster fraction to 70%. Better, but
"far from center" catches both genuine connectors and points that are just
unusual within their own topic.

Multi-address detection asks a different question: no matter how we divide the
data — coarse or fine — does this point keep landing on the boundary? Points
that are boundary at multiple levels of the hierarchy have 90% of their nearest
neighbors in a different cluster. They're not outliers — they're connectors,
the points that link otherwise separate regions of meaning.

Usage:
    python demo/bridge_comparison.py demo/wiki_simple_50k.parquet --sample 10000
"""

import argparse
import time

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from dyf import cut_tree_to_labels
from dyf.pca_tree import build_pca_tree, extract_boundary_persistence


def print_table(title, headers, rows):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # Compute column widths
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
        description="Compare simple LSH bridges vs multi-address PCA tree bridges")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12,
                        help="PCA tree depth")
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Number of clusters for ground-truth labels")
    parser.add_argument("--margin-pct", type=float, default=0.10,
                        help="Margin percentile threshold for multi-address")
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

    titles = [t for t, keep in zip(titles_all, dedup_mask) if keep]
    embeddings = embeddings_all[dedup_mask]
    n = len(titles)
    print(f"  {len(titles_all)} -> {n} after dedup")

    # ── Simple LSH bridges: low centroid similarity ──────────────────────
    print("\nComputing simple LSH bridge scores (centroid similarity)...")
    # Refit on deduped data for consistent indexing
    clf2 = RustClassifier(
        embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf2.fit(embeddings)
    centroid_sims = clf2.get_centroid_similarities()
    # Rank ascending = lowest similarity first = strongest bridges
    simple_rank = np.argsort(centroid_sims)

    # ── Multi-address PCA tree bridges ───────────────────────────────────
    print(f"\nBuilding PCA tree (depth={args.max_depth})...")
    t0 = time.time()
    tree = build_pca_tree(embeddings, args.max_depth)
    print(f"  Built in {time.time() - t0:.1f}s")

    print(f"Extracting multi-address points (margin_pct={args.margin_pct})...")
    ma_result = extract_boundary_persistence(tree, margin_pct=args.margin_pct)
    boundary_count = ma_result['boundary_count']
    boundary_depths = ma_result['boundary_depths']

    # Depth-weighted score: boundary at depth 0 (root) is worth max_depth,
    # boundary at deepest level is worth 1. Being boundary where the split
    # divides the whole dataset matters far more than near a tiny leaf.
    ma_score = np.zeros(n, dtype=np.float64)
    for pt_idx, depths in boundary_depths.items():
        ma_score[pt_idx] = sum(args.max_depth - d for d in depths)

    # Rank descending = highest weighted score first = strongest bridges
    # Break ties with lower centroid_sim (more bridge-like)
    ma_rank = np.lexsort((centroid_sims, -ma_score))

    bc_nonzero = np.sum(boundary_count > 0)
    bc_multi = np.sum(boundary_count > 1)
    print(f"  Boundary at >=1 depth: {bc_nonzero}/{n} ({100*bc_nonzero/n:.1f}%)")
    print(f"  Boundary at >=2 depths: {bc_multi}/{n} ({100*bc_multi/n:.1f}%)")
    print(f"  Max boundary count: {boundary_count.max()}")
    print(f"  Weighted score: median={np.median(ma_score):.1f}, "
          f"max={ma_score.max():.1f}, "
          f"top-100 min={np.sort(ma_score)[::-1][99]:.1f}")

    # ── Ground-truth cluster labels ──────────────────────────────────────
    print(f"\nCutting PCA tree to {args.n_clusters} clusters for ground-truth labels...")
    cluster_labels = cut_tree_to_labels(
        tree, n, args.n_clusters, max_depth=args.max_depth)
    n_actual = len(set(cluster_labels))
    print(f"  Got {n_actual} clusters")

    # ── Brute-force kNN in high-D cosine ─────────────────────────────────
    print("\nComputing brute-force k=20 nearest neighbors (cosine)...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=21, metric='cosine')
    nn.fit(embeddings)
    _, knn_indices = nn.kneighbors(embeddings)
    knn_indices = knn_indices[:, 1:]  # drop self
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Random baseline indices ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    random_rank = rng.permutation(n)

    # ── Topic Diversity ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Topic Diversity")
    print("  (mean distinct clusters among k=20 high-D nearest neighbors)")
    print("=" * 70)

    # Precompute per-point metrics for all points
    diversity_all = np.zeros(n, dtype=int)
    cross_frac_all = np.zeros(n, dtype=np.float64)
    for pt in range(n):
        neighbors = knn_indices[pt]
        neighbor_labels = cluster_labels[neighbors]
        diversity_all[pt] = len(set(neighbor_labels.tolist()))
        cross_frac_all[pt] = np.mean(neighbor_labels != cluster_labels[pt])

    # Report population baselines
    print(f"\n  Population baseline (all {n} points):")
    print(f"    Diversity:  mean={diversity_all.mean():.2f}, "
          f"median={np.median(diversity_all):.0f}")
    print(f"    Cross-frac: mean={cross_frac_all.mean():.3f}, "
          f"median={np.median(cross_frac_all):.3f}")

    budgets = [100, 250, 500, 1000, 2000]

    # ── Table 1: Topic Diversity ──
    rows_diversity = []
    for budget in budgets:
        results = {}
        for name, ranked_indices in [("Simple LSH", simple_rank),
                                      ("Multi-Addr", ma_rank),
                                      ("Random", random_rank)]:
            selected = ranked_indices[:budget]
            mean_div = diversity_all[selected].mean()
            median_div = np.median(diversity_all[selected])
            results[name] = (mean_div, median_div)

        rows_diversity.append([
            str(budget),
            f"{results['Simple LSH'][0]:.2f} ({results['Simple LSH'][1]:.0f})",
            f"{results['Multi-Addr'][0]:.2f} ({results['Multi-Addr'][1]:.0f})",
            f"{results['Random'][0]:.2f} ({results['Random'][1]:.0f})",
        ])

    print_table(
        "Topic Diversity: mean (median) distinct clusters per bridge point",
        ["Budget", "Simple LSH", "Multi-Address", "Random"],
        rows_diversity,
    )

    # ── Table 2: Cross-Cluster Neighbor Fraction ──
    rows_cross = []
    for budget in budgets:
        results = {}
        for name, ranked_indices in [("Simple LSH", simple_rank),
                                      ("Multi-Addr", ma_rank),
                                      ("Random", random_rank)]:
            selected = ranked_indices[:budget]
            mean_cf = cross_frac_all[selected].mean()
            median_cf = np.median(cross_frac_all[selected])
            results[name] = (mean_cf, median_cf)

        rows_cross.append([
            str(budget),
            f"{results['Simple LSH'][0]:.3f} ({results['Simple LSH'][1]:.3f})",
            f"{results['Multi-Addr'][0]:.3f} ({results['Multi-Addr'][1]:.3f})",
            f"{results['Random'][0]:.3f} ({results['Random'][1]:.3f})",
        ])

    print_table(
        "Cross-Cluster Neighbor Fraction: mean (median)",
        ["Budget", "Simple LSH", "Multi-Address", "Random"],
        rows_cross,
    )

    # ── Summary stats ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERLAP ANALYSIS")
    print("=" * 70)

    for budget in [500, 1000]:
        simple_set = set(simple_rank[:budget].tolist())
        ma_set = set(ma_rank[:budget].tolist())
        overlap = len(simple_set & ma_set)
        print(f"  Budget {budget}: overlap = {overlap}/{budget} "
              f"({100*overlap/budget:.1f}%)")

    print()


if __name__ == "__main__":
    main()
