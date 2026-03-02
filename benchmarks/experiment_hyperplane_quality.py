"""Hyperplane quality analysis: what does PCA alignment actually give us?

Measures:
1. Variance explained by PCA hyperplanes
2. Projection magnitude distribution (all items, not just queries)
3. Neighbor co-location rate: what fraction of an item's k-NN share its bucket?
4. Projection magnitude vs co-location: do items far from boundaries have
   better neighbor co-location?
5. Cluster-splitting: do hyperplanes separate clusters or cut through them?
6. Comparison: PCA vs ITQ vs random hyperplanes
"""

import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import PCA


N_ITEMS = 100_000
EMBEDDING_DIM = 128
N_CLUSTERS = 100
K = 10
SEED = 42


def generate_clustered_data(n_items, dim, n_clusters, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assignments = rng.integers(0, n_clusters, size=n_items)
    noise = rng.standard_normal((n_items, dim)).astype(np.float32) * 0.15
    embeddings = centers[assignments] + noise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)
    return embeddings, assignments


def main():
    print("=== Hyperplane Quality Analysis ===\n")

    embeddings, assignments = generate_clustered_data(
        N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)
    print(f"{N_ITEMS} items, {EMBEDDING_DIM}d, {N_CLUSTERS} clusters\n")

    # ---------------------------------------------------------------
    # 1. PCA variance explained
    # ---------------------------------------------------------------
    print("--- 1. PCA Variance Explained ---")
    pca = PCA(n_components=min(30, EMBEDDING_DIM))
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for n in [1, 2, 3, 5, 9, 12, 15, 20, 30]:
        if n <= len(cumvar):
            print(f"  Top {n:>2} PCs: {cumvar[n-1]*100:>5.1f}% variance")
    print()

    # ---------------------------------------------------------------
    # 2. DensityClassifier hyperplanes vs raw PCA
    # ---------------------------------------------------------------
    from dyf_rs import DensityClassifier

    num_bits = 9
    clf = DensityClassifier(embedding_dim=EMBEDDING_DIM, num_bits=num_bits,
                            seed=SEED, skip_isolation=True)
    clf.fit(embeddings)
    dc_hp = np.array(clf.get_hyperplanes(), dtype=np.float32)  # (num_bits, dim)
    dc_bids = np.array(clf.get_bucket_ids())

    # Raw PCA directions for comparison
    pca_hp = pca.components_[:num_bits].astype(np.float32)

    # Random hyperplanes
    rng = np.random.default_rng(SEED)
    rand_hp = rng.standard_normal((num_bits, EMBEDDING_DIM)).astype(np.float32)
    rand_hp /= np.linalg.norm(rand_hp, axis=1, keepdims=True)

    # ---------------------------------------------------------------
    # 3. Projection magnitude distributions
    # ---------------------------------------------------------------
    print("--- 2. Projection Magnitude Distribution ---")
    for name, hp in [("DensityClf", dc_hp), ("Raw PCA", pca_hp), ("Random", rand_hp)]:
        projections = embeddings @ hp.T  # (N, num_bits)
        abs_proj = np.abs(projections)

        # Per-bit stats
        print(f"\n  {name} hyperplanes ({num_bits} bits):")
        print(f"  {'Bit':>4} {'Mean|proj|':>11} {'Std':>8} {'p10':>8} "
              f"{'p50':>8} {'p90':>8}")
        for i in range(num_bits):
            vals = abs_proj[:, i]
            print(f"  {i:>4} {vals.mean():>11.5f} {vals.std():>8.5f} "
                  f"{np.percentile(vals, 10):>8.5f} "
                  f"{np.percentile(vals, 50):>8.5f} "
                  f"{np.percentile(vals, 90):>8.5f}")

        # Overall: min margin per item (the "weakest link" bit)
        min_margins = abs_proj.min(axis=1)
        print(f"  Min margin per item: mean={min_margins.mean():.5f}, "
              f"p10={np.percentile(min_margins, 10):.5f}, "
              f"p50={np.percentile(min_margins, 50):.5f}")

    # ---------------------------------------------------------------
    # 4. Neighbor co-location rate
    # ---------------------------------------------------------------
    print("\n--- 3. Neighbor Co-location Rate ---")
    print("  (What fraction of an item's 10-NN share its LSH bucket?)\n")

    # Sample items for NN computation
    rng2 = np.random.default_rng(SEED + 1)
    sample_idx = rng2.choice(N_ITEMS, size=2000, replace=False)

    # Brute-force 10-NN for sample
    sample_emb = embeddings[sample_idx]
    sims = sample_emb @ embeddings.T  # (2000, N)

    for name, hp in [("DensityClf", dc_hp), ("Raw PCA", pca_hp), ("Random", rand_hp)]:
        # Compute bucket IDs for all items
        projections = embeddings @ hp.T
        bits = (projections > 0).astype(np.uint64)
        bucket_ids = np.zeros(N_ITEMS, dtype=np.int64)
        for i in range(num_bits):
            bucket_ids += bits[:, i].astype(np.int64) << i

        coloc_rates = []
        for i, si in enumerate(sample_idx):
            # Get 10-NN (excluding self)
            item_sims = sims[i].copy()
            item_sims[si] = -np.inf
            nn_idx = np.argpartition(-item_sims, K)[:K]

            # Co-location rate
            item_bid = bucket_ids[si]
            nn_bids = bucket_ids[nn_idx]
            coloc = np.mean(nn_bids == item_bid)
            coloc_rates.append(coloc)

        coloc_rates = np.array(coloc_rates)
        print(f"  {name:>12}: mean={coloc_rates.mean():.4f}, "
              f"p10={np.percentile(coloc_rates, 10):.3f}, "
              f"p50={np.percentile(coloc_rates, 50):.3f}, "
              f"p90={np.percentile(coloc_rates, 90):.3f}")

    # ---------------------------------------------------------------
    # 5. Co-location vs min margin (is being far from boundary helpful?)
    # ---------------------------------------------------------------
    print("\n--- 4. Co-location vs Projection Margin ---")
    print("  (Do items far from ALL hyperplane boundaries have better "
          "neighbor co-location?)\n")

    # Use DensityClf hyperplanes
    dc_projections = embeddings @ dc_hp.T
    dc_abs_proj = np.abs(dc_projections)
    dc_min_margins = dc_abs_proj.min(axis=1)
    dc_mean_margins = dc_abs_proj.mean(axis=1)

    dc_bits = (dc_projections > 0).astype(np.uint64)
    dc_bucket_ids = np.zeros(N_ITEMS, dtype=np.int64)
    for i in range(num_bits):
        dc_bucket_ids += dc_bits[:, i].astype(np.int64) << i

    # Compute co-location for sample, binned by min margin
    margin_coloc = []
    for i, si in enumerate(sample_idx):
        item_sims = sims[i].copy()
        item_sims[si] = -np.inf
        nn_idx = np.argpartition(-item_sims, K)[:K]
        coloc = np.mean(dc_bucket_ids[nn_idx] == dc_bucket_ids[si])
        margin_coloc.append((dc_min_margins[si], dc_mean_margins[si], coloc))

    margin_coloc = np.array(margin_coloc)
    # Bin by min margin quintiles
    quintiles = np.percentile(margin_coloc[:, 0], [0, 20, 40, 60, 80, 100])
    print(f"  {'Min margin bin':>20} {'Mean coloc':>11} {'Count':>7}")
    for q in range(5):
        lo, hi = quintiles[q], quintiles[q + 1]
        mask = (margin_coloc[:, 0] >= lo) & (margin_coloc[:, 0] < hi + 1e-9)
        if mask.sum() > 0:
            mean_coloc = margin_coloc[mask, 2].mean()
            print(f"  [{lo:.4f}, {hi:.4f}){'>':>2} {mean_coloc:>11.4f} "
                  f"{mask.sum():>7}")

    # Also bin by mean margin
    print()
    quintiles_m = np.percentile(margin_coloc[:, 1], [0, 20, 40, 60, 80, 100])
    print(f"  {'Mean margin bin':>20} {'Mean coloc':>11} {'Count':>7}")
    for q in range(5):
        lo, hi = quintiles_m[q], quintiles_m[q + 1]
        mask = (margin_coloc[:, 1] >= lo) & (margin_coloc[:, 1] < hi + 1e-9)
        if mask.sum() > 0:
            mean_coloc = margin_coloc[mask, 2].mean()
            print(f"  [{lo:.4f}, {hi:.4f}){'>':>2} {mean_coloc:>11.4f} "
                  f"{mask.sum():>7}")

    # ---------------------------------------------------------------
    # 6. Cluster splitting by hyperplanes
    # ---------------------------------------------------------------
    print("\n--- 5. Cluster Splitting ---")
    print("  (How many LSH buckets does each true cluster span?)\n")

    for name, hp in [("DensityClf", dc_hp), ("Raw PCA", pca_hp), ("Random", rand_hp)]:
        projections = embeddings @ hp.T
        bits = (projections > 0).astype(np.uint64)
        bucket_ids = np.zeros(N_ITEMS, dtype=np.int64)
        for i in range(num_bits):
            bucket_ids += bits[:, i].astype(np.int64) << i

        buckets_per_cluster = []
        for c in range(N_CLUSTERS):
            mask = assignments == c
            cluster_bids = set(bucket_ids[mask].tolist())
            buckets_per_cluster.append(len(cluster_bids))

        bpc = np.array(buckets_per_cluster)
        print(f"  {name:>12}: mean={bpc.mean():.1f}, "
              f"min={bpc.min()}, p50={np.median(bpc):.0f}, "
              f"max={bpc.max()} buckets per cluster")

    # ---------------------------------------------------------------
    # 7. Alignment between PCA hyperplanes and DensityClf hyperplanes
    # ---------------------------------------------------------------
    print("\n--- 6. Hyperplane Alignment ---")
    print("  (Cosine similarity between DensityClf and raw PCA directions)\n")

    # Normalize all hyperplanes
    dc_hp_n = dc_hp / np.linalg.norm(dc_hp, axis=1, keepdims=True)
    pca_hp_n = pca_hp / np.linalg.norm(pca_hp, axis=1, keepdims=True)

    sim_matrix = np.abs(dc_hp_n @ pca_hp_n.T)
    print("  |cos| similarity (DensityClf row, PCA col):")
    print(f"  {'':>6}", end="")
    for j in range(num_bits):
        print(f" PC{j:>2}", end="")
    print()
    for i in range(num_bits):
        print(f"  DC{i:>2}:", end="")
        for j in range(num_bits):
            print(f" {sim_matrix[i, j]:.2f}", end="")
        print()

    # Best PCA match per DC hyperplane
    print(f"\n  Best PCA match per DensityClf hyperplane:")
    for i in range(num_bits):
        best_j = np.argmax(sim_matrix[i])
        print(f"    DC{i} -> PC{best_j} (|cos|={sim_matrix[i, best_j]:.4f})")

    # ---------------------------------------------------------------
    # 8. What about fewer bits? Compare co-location at different bit counts
    # ---------------------------------------------------------------
    print("\n--- 7. Co-location Rate vs Bit Count ---")
    print("  (Using DensityClassifier at each bit count)\n")

    for nb in [1, 2, 3, 4, 5, 6, 9, 12]:
        clf_nb = DensityClassifier(embedding_dim=EMBEDDING_DIM, num_bits=nb,
                                   seed=SEED, skip_isolation=True)
        clf_nb.fit(embeddings)
        bids_nb = np.array(clf_nb.get_bucket_ids())
        n_buckets = len(np.unique(bids_nb))

        coloc_rates = []
        for i, si in enumerate(sample_idx):
            item_sims = sims[i].copy()
            item_sims[si] = -np.inf
            nn_idx = np.argpartition(-item_sims, K)[:K]
            coloc = np.mean(bids_nb[nn_idx] == bids_nb[si])
            coloc_rates.append(coloc)

        coloc_rates = np.array(coloc_rates)
        # Expected co-location if random: 1/n_buckets * K / N * bucket_size ≈ 1/n_buckets
        expected_random = 1.0 / n_buckets
        lift = coloc_rates.mean() / expected_random

        print(f"  {nb:>2} bits ({n_buckets:>5} bkts): "
              f"coloc={coloc_rates.mean():.4f}, "
              f"random={expected_random:.4f}, "
              f"lift={lift:.2f}x")


if __name__ == '__main__':
    main()
