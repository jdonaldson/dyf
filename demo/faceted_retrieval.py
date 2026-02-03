"""
Faceted Retrieval: Bridge-Boosted Result Diversification

Standard vector search returns the k nearest embeddings. For most queries,
those cluster in one or two topics — missing relevant documents from other
facets of the query. RAG systems need broad coverage, not just proximity.

MMR (Maximal Marginal Relevance) is the standard fix: iteratively pick
results that are both relevant and dissimilar to already-selected results.
But MMR uses geometric distance as its diversity signal, which doesn't
distinguish "different topic" from "same topic, slightly different phrasing."

Multi-address bridges offer a structural alternative. They sit on actual
topic boundaries, so promoting them in re-ranking should improve topic
coverage — getting results from more distinct semantic regions, not just
pushing results apart in embedding space.

Re-ranking strategies (all operate on a shared top-100 candidate pool):
  1. Standard kNN:    take the 10 nearest vectors
  2. MMR:             iteratively select for relevance + diversity
  3. Bridge-boost:    blend similarity with boundary persistence bridge score
  4. Bridge+MMR:      MMR where diversity is bridge-weighted

Metrics:
  - Topic coverage:   distinct clusters in top-10
  - Mean relevance:   average cosine similarity of top-10 to query
  - Coverage@cost:    coverage gained per unit of relevance sacrificed

Usage:
    python demo/faceted_retrieval.py demo/wiki_simple_50k.parquet --sample 10000
"""

import argparse
import time

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from dyf.pca_tree import (
    build_pca_tree,
    cut_tree_to_labels,
    extract_boundary_persistence,
)
from dyf.rerank import (
    rerank_bridge_boost,
    rerank_bridge_mmr,
    rerank_mmr,
    rerank_standard,
)


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
        description="Faceted retrieval: bridge-boosted result diversification")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Number of topic clusters")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--candidate-pool", type=int, default=100,
                        help="Size of initial candidate pool per query")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Final result set size")
    args = parser.parse_args()

    # ── Load & dedup ─────────────────────────────────────────────────────
    print(f"Loading {args.parquet_path}...")
    df = pl.read_parquet(args.parquet_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, seed=42)

    titles_all = df["title"].to_list()
    embeddings_all = np.array(df["embedding"].to_list(), dtype=np.float32)

    from dyf.chunks import deduplicate_chunks, neighbor_coherence
    from dyf_rs import DensityClassifier as RustClassifier

    clf = RustClassifier(
        embedding_dim=embeddings_all.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings_all)
    bucket_ids = np.asarray(clf.get_bucket_ids())
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles_all))

    titles = [t for t, keep in zip(titles_all, dedup_mask) if keep]
    embeddings = embeddings_all[dedup_mask]
    n = len(embeddings)
    print(f"  {len(titles_all)} -> {n} after dedup")

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

    n_bridge = np.sum(ma_score > 0)
    print(f"  Bridge points: {n_bridge}/{n} ({100*n_bridge/n:.1f}%)")
    print(f"  Score: median={np.median(ma_score[ma_score > 0]):.1f}, "
          f"max={ma_score.max():.1f}")

    # ── Cluster assignment ───────────────────────────────────────────────
    print(f"\nCutting PCA tree to {args.n_clusters} clusters...")
    cluster_labels = cut_tree_to_labels(
        tree, args.max_depth, n, args.n_clusters)
    n_actual = len(set(cluster_labels.tolist()))
    print(f"  Got {n_actual} clusters")

    # Remap to contiguous
    unique_labels = sorted(set(cluster_labels.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels = np.array([label_map[c] for c in cluster_labels], dtype=int)

    sizes = np.bincount(cluster_labels)
    print(f"  Cluster sizes: min={sizes.min()}, median={int(np.median(sizes))}, "
          f"max={sizes.max()}")

    # ── Candidate pool: brute-force kNN ──────────────────────────────────
    pool_k = args.candidate_pool
    print(f"\nComputing brute-force k={pool_k} nearest neighbors...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=pool_k + 1, metric='cosine')
    nn.fit(embeddings)
    pool_dists, pool_indices = nn.kneighbors(embeddings)
    pool_dists = pool_dists[:, 1:]     # drop self
    pool_indices = pool_indices[:, 1:]
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Neighbor coherence → meta cluster detection ──────────────────────
    print("\nComputing neighbor coherence (k=100)...")
    t0 = time.time()
    coh = neighbor_coherence(embeddings, pool_indices, sample_k=100)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Per-cluster mean coherence → flag meta clusters
    cluster_mean_coh = np.zeros(n_actual)
    for c in range(n_actual):
        mask = cluster_labels == c
        cluster_mean_coh[c] = coh[mask].mean()

    # Meta threshold: clusters whose mean coherence is above the 75th
    # percentile of cluster coherences are structural echo chambers.
    coh_threshold = np.percentile(cluster_mean_coh, 75)
    meta_clusters = set()
    print(f"\n  Cluster coherence (threshold={coh_threshold:.4f}):")
    for c in np.argsort(-cluster_mean_coh):
        c_size = int(np.sum(cluster_labels == c))
        c_titles = [titles[i] for i in range(n) if cluster_labels[i] == c]
        sample = " | ".join(t[:18] for t in c_titles[:3])
        is_meta = cluster_mean_coh[c] > coh_threshold
        if is_meta:
            meta_clusters.add(c)
        flag = " META" if is_meta else "     "
        print(f"    [{flag}] cluster {c:>2d}: coh={cluster_mean_coh[c]:.4f}  "
              f"n={c_size:>4d}  {sample}")

    print(f"\n  Meta clusters: {len(meta_clusters)}/{n_actual} "
          f"({100*len(meta_clusters)/n_actual:.0f}%)")
    n_meta_points = sum(1 for i in range(n) if cluster_labels[i] in meta_clusters)
    print(f"  Meta points: {n_meta_points}/{n} ({100*n_meta_points/n:.1f}%)")

    # Convert distances to similarities (sklearn cosine distance = 1 - cos_sim)
    pool_sims = 1.0 - pool_dists

    # ── Select query points ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_queries = min(args.n_queries, n)
    query_indices = rng.choice(n, n_queries, replace=False)
    top_k = args.top_k

    # ── Evaluate strategies ──────────────────────────────────────────────
    print(f"\nEvaluating {n_queries} queries, pool={pool_k}, top_k={top_k}...")

    strategy_names = ["Standard kNN", "MMR (λ=0.5)", "MMR (λ=0.7)",
                      "Bridge-boost", "Bridge+MMR", "Bridge+MMR+Meta"]

    # Per-query metrics
    all_coverage = {s: [] for s in strategy_names}
    all_topical_cov = {s: [] for s in strategy_names}  # excluding meta clusters
    all_relevance = {s: [] for s in strategy_names}
    all_results_clusters = {s: [] for s in strategy_names}

    t0 = time.time()
    for qi, q in enumerate(query_indices):
        q_emb = emb_normed[q]
        cand_idx = pool_indices[q]
        cand_sims = pool_sims[q]

        for strategy in strategy_names:
            if strategy == "Standard kNN":
                selected = rerank_standard(cand_sims, cand_idx, top_k)
            elif strategy == "MMR (λ=0.5)":
                selected = rerank_mmr(q_emb, cand_idx, emb_normed, top_k,
                                       lam=0.5)
            elif strategy == "MMR (λ=0.7)":
                selected = rerank_mmr(q_emb, cand_idx, emb_normed, top_k,
                                       lam=0.7)
            elif strategy == "Bridge-boost":
                selected = rerank_bridge_boost(
                    cand_sims, cand_idx, ma_score, top_k, alpha=0.3)
            elif strategy == "Bridge+MMR":
                selected = rerank_bridge_mmr(
                    q_emb, cand_idx, emb_normed, ma_score,
                    cluster_labels, top_k, lam=0.5)
            elif strategy == "Bridge+MMR+Meta":
                selected = rerank_bridge_mmr(
                    q_emb, cand_idx, emb_normed, ma_score,
                    cluster_labels, top_k, lam=0.5,
                    meta_clusters=meta_clusters)

            # Metrics
            result_clusters = set(cluster_labels[selected].tolist())
            coverage = len(result_clusters)
            topical_cov = len(result_clusters - meta_clusters)
            result_sims = emb_normed[selected] @ q_emb
            relevance = float(np.mean(result_sims))

            all_coverage[strategy].append(coverage)
            all_topical_cov[strategy].append(topical_cov)
            all_relevance[strategy].append(relevance)
            all_results_clusters[strategy].append(result_clusters)

        if (qi + 1) % 200 == 0:
            print(f"  {qi + 1}/{n_queries}...")

    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Summary table ────────────────────────────────────────────────────
    rows = []
    for strategy in strategy_names:
        cov = np.array(all_coverage[strategy])
        tcov = np.array(all_topical_cov[strategy])
        rel = np.array(all_relevance[strategy])
        rows.append([
            strategy,
            f"{cov.mean():.2f}",
            f"{tcov.mean():.2f}",
            f"{rel.mean():.4f}",
        ])

    n_topical = n_actual - len(meta_clusters)
    print_table(
        f"Faceted Retrieval: top-{top_k} from pool of {pool_k} "
        f"({n_actual} clusters, {n_topical} topical)",
        ["Strategy", "All Cov", "Topic Cov", "Relevance"],
        rows,
    )

    # ── Coverage distribution ────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Coverage distribution (% of queries hitting N+ clusters)")
    print(f"{'=' * 78}")

    thresholds = [1, 2, 3, 4, 5, 6, 7, 8]
    dist_rows = []
    for threshold in thresholds:
        row = [f">={threshold}"]
        for strategy in strategy_names:
            cov = np.array(all_coverage[strategy])
            pct = 100 * np.mean(cov >= threshold)
            row.append(f"{pct:.1f}%")
        dist_rows.append(row)

    print_table(
        "% of queries with coverage >= threshold",
        ["Clusters"] + strategy_names,
        dist_rows,
    )

    # ── Relevance-coverage tradeoff ──────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Relevance vs Topical Coverage Tradeoff")
    print(f"{'=' * 78}")

    baseline_tcov = np.mean(all_topical_cov["Standard kNN"])
    baseline_rel = np.mean(all_relevance["Standard kNN"])

    print(f"\n  Baseline (Standard kNN): topical_cov={baseline_tcov:.2f}, "
          f"relevance={baseline_rel:.4f}")

    for strategy in strategy_names:
        if strategy == "Standard kNN":
            continue
        tcov = np.mean(all_topical_cov[strategy])
        rel = np.mean(all_relevance[strategy])
        tcov_gain = tcov - baseline_tcov
        rel_cost = baseline_rel - rel
        if rel_cost > 0:
            ratio = tcov_gain / rel_cost
            print(f"  {strategy:20s}: topical +{tcov_gain:.2f}, "
                  f"relevance -{rel_cost:.4f}, "
                  f"ratio={ratio:.1f} topics per 0.001 rel")
        elif tcov_gain > 0:
            print(f"  {strategy:20s}: topical +{tcov_gain:.2f}, "
                  f"relevance +{-rel_cost:.4f} (free lunch!)")
        else:
            print(f"  {strategy:20s}: topical {tcov_gain:+.2f}, "
                  f"relevance {-rel_cost:+.4f}")

    # ── Stratify by query type ───────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Stratified analysis: boundary vs core queries")
    print(f"{'=' * 78}")

    # A query is "boundary" if its true nearest neighbors span many clusters
    true_k10 = pool_indices[:, :10]
    query_topic_span = np.zeros(n_queries)
    for qi, q in enumerate(query_indices):
        nb_clusters = cluster_labels[true_k10[q]]
        query_topic_span[qi] = len(set(nb_clusters.tolist()))

    boundary_mask = query_topic_span >= np.percentile(query_topic_span, 75)
    core_mask = query_topic_span <= np.percentile(query_topic_span, 25)

    for label, mask in [("Boundary queries", boundary_mask),
                         ("Core queries", core_mask)]:
        if mask.sum() == 0:
            continue
        print(f"\n  {label} (n={mask.sum()}, "
              f"avg topic span={query_topic_span[mask].mean():.1f}):")
        for strategy in strategy_names:
            tcov = np.array(all_topical_cov[strategy])[mask]
            rel = np.array(all_relevance[strategy])[mask]
            print(f"    {strategy:20s}: topical_cov={tcov.mean():.2f}  "
                  f"relevance={rel.mean():.4f}")

    # ── Example queries ──────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Example queries: Bridge+MMR+Meta vs Standard kNN")
    print(f"{'=' * 78}")

    tcov_standard = np.array(all_topical_cov["Standard kNN"])
    tcov_meta_mmr = np.array(all_topical_cov["Bridge+MMR+Meta"])
    improvement = tcov_meta_mmr - tcov_standard

    # Show top 5 improvements
    top_improved = np.argsort(-improvement)[:5]
    for rank, qi in enumerate(top_improved):
        q = query_indices[qi]
        std_tc = tcov_standard[qi]
        bm_tc = tcov_meta_mmr[qi]
        if bm_tc <= std_tc:
            break
        print(f"\n  #{rank+1}: \"{titles[q][:60]}\"")
        print(f"    Standard: {int(std_tc)} topical clusters, "
              f"Bridge+MMR+Meta: {int(bm_tc)} (+{int(bm_tc - std_tc)})")

        # Show what was found
        bm_results = rerank_bridge_mmr(
            emb_normed[q], pool_indices[q], emb_normed, ma_score,
            cluster_labels, top_k, lam=0.5,
            meta_clusters=meta_clusters)
        std_results = rerank_standard(
            pool_sims[q], pool_indices[q], top_k)

        bm_cls = cluster_labels[bm_results]
        std_cls = cluster_labels[std_results]
        new_topical = (set(bm_cls.tolist()) - set(std_cls.tolist())) - meta_clusters
        if new_topical:
            for c in sorted(new_topical):
                c_results = bm_results[bm_cls == c]
                if len(c_results) > 0:
                    t = titles[c_results[0]][:50]
                    print(f"    + topic {c}: \"{t}\"")

    print()


if __name__ == "__main__":
    main()
