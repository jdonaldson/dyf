"""Direct comparison: raw PCA hyperplanes vs DensityClassifier's centroid PCA.

Measures recall@10 (reachable GT items) at each nprobe level for:
1. DensityClassifier (centroid PCA)
2. Raw PCA on full data
3. ITQ (raw PCA + rotation)
4. Random hyperplanes (baseline)

Also tests multi-table variants for the best hyperplane type.
"""

import numpy as np
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


def brute_force_topk(embeddings, query, k):
    sims = embeddings @ query
    topk = np.argpartition(-sims, k)[:k]
    topk = topk[np.argsort(-sims[topk])]
    return topk, sims[topk]


def hash_all(embeddings, hyperplanes):
    """Hash all items using sign of projection."""
    num_bits = hyperplanes.shape[0]
    projections = embeddings @ hyperplanes.T  # (N, num_bits)
    bits = (projections > 0).astype(np.int64)
    bucket_ids = np.zeros(len(embeddings), dtype=np.int64)
    for i in range(num_bits):
        bucket_ids += bits[:, i] << i
    return bucket_ids, projections


def hash_query_with_proj(query, hyperplanes):
    projections = hyperplanes @ query
    bits = (projections > 0).astype(np.uint64)
    bucket_id = np.uint64(0)
    for i, b in enumerate(bits):
        bucket_id |= (b << np.uint64(i))
    return int(bucket_id), projections


def margin_distance(a, b, projections, num_bits):
    xor = a ^ b
    cost = 0.0
    for i in range(num_bits):
        if xor & (1 << i):
            cost += abs(float(projections[i]))
    return cost


def compute_recall_curve(embeddings, bucket_ids, hyperplanes, num_bits,
                          query_indices, gt_all, nprobe_values):
    """For each nprobe, compute recall@10 (fraction of GT reachable)."""
    unique_bids = np.unique(bucket_ids)

    recalls = {nv: [] for nv in nprobe_values}

    for qi in query_indices:
        query = embeddings[qi]
        gt_set = gt_all[qi]
        query_bid, projections = hash_query_with_proj(query, hyperplanes)

        # Rank buckets by margin distance
        bid_costs = []
        for bid in unique_bids:
            bid_int = int(bid)
            if bid_int == query_bid:
                bid_costs.append((0.0, bid_int))
            else:
                cost = margin_distance(query_bid, bid_int, projections, num_bits)
                bid_costs.append((cost, bid_int))
        bid_costs.sort()

        # For each nprobe level, count reachable GT items
        for nv in nprobe_values:
            probe_bids = set(bid for _, bid in bid_costs[:nv])
            # Count GT items in probed buckets
            found = sum(1 for gi in gt_set if int(bucket_ids[gi]) in probe_bids)
            recalls[nv].append(found / K)

    return {nv: float(np.mean(vals)) for nv, vals in recalls.items()}


def itq_rotation(embeddings, hyperplanes, n_iter=50):
    """Iterative Quantization: rotate PCA directions to minimize quantization error."""
    projections = embeddings @ hyperplanes.T  # (N, num_bits)
    num_bits = hyperplanes.shape[0]

    R = np.eye(num_bits, dtype=np.float32)

    for _ in range(n_iter):
        # Quantize
        rotated = projections @ R
        B = np.sign(rotated)
        B[B == 0] = 1

        # SVD to find best rotation
        U, _, Vt = np.linalg.svd(B.T @ projections @ R)
        R = (Vt.T @ U.T).astype(np.float32)

    # Rotated hyperplanes
    rotated_hp = (R.T @ hyperplanes).astype(np.float32)
    # Re-normalize rows
    norms = np.linalg.norm(rotated_hp, axis=1, keepdims=True)
    rotated_hp /= np.maximum(norms, 1e-10)
    return rotated_hp


def main():
    print("=== PCA vs Centroid-PCA Hyperplane Comparison ===\n")

    embeddings, assignments = generate_clustered_data(
        N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)
    print(f"{N_ITEMS} items, {EMBEDDING_DIM}d, {N_CLUSTERS} clusters\n")

    num_bits = 9
    nprobe_values = [1, 2, 3, 5, 10, 20, 50, 100]

    # Pre-compute ground truth
    rng = np.random.default_rng(SEED + 1)
    n_queries = 1000
    query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

    gt_all = {}
    for qi in query_indices:
        gt_idx, _ = brute_force_topk(embeddings, embeddings[qi], K)
        gt_all[qi] = set(int(x) for x in gt_idx)

    # --- Build hyperplane sets ---

    # 1. DensityClassifier (centroid PCA)
    from dyf_rs import DensityClassifier
    clf = DensityClassifier(embedding_dim=EMBEDDING_DIM, num_bits=num_bits,
                            seed=SEED, skip_isolation=True)
    clf.fit(embeddings)
    dc_hp = np.array(clf.get_hyperplanes(), dtype=np.float32)
    dc_bids = np.array(clf.get_bucket_ids())

    # 2. Raw PCA
    pca = PCA(n_components=num_bits)
    pca.fit(embeddings)
    pca_hp = pca.components_.astype(np.float32)
    pca_bids, _ = hash_all(embeddings, pca_hp)

    # 3. ITQ (PCA + rotation)
    itq_hp = itq_rotation(embeddings, pca_hp, n_iter=50)
    itq_bids, _ = hash_all(embeddings, itq_hp)

    # 4. Random
    rng_hp = np.random.default_rng(SEED)
    rand_hp = rng_hp.standard_normal((num_bits, EMBEDDING_DIM)).astype(np.float32)
    rand_hp /= np.linalg.norm(rand_hp, axis=1, keepdims=True)
    rand_bids, _ = hash_all(embeddings, rand_hp)

    methods = [
        ("CentroidPCA", dc_hp, dc_bids),
        ("RawPCA", pca_hp, pca_bids),
        ("ITQ", itq_hp, itq_bids),
        ("Random", rand_hp, rand_bids),
    ]

    # --- Quick stats ---
    print("Bucket distribution:")
    for name, hp, bids in methods:
        n_buckets = len(np.unique(bids))
        sizes = np.bincount(bids.astype(np.int64))
        sizes = sizes[sizes > 0]
        print(f"  {name:>14}: {n_buckets:>4} buckets, "
              f"sizes min={sizes.min()} med={int(np.median(sizes))} "
              f"max={sizes.max()}")
    print()

    # --- Co-location ---
    print("Neighbor co-location (10-NN, 2000 sample):")
    sample_idx = rng.choice(N_ITEMS, size=2000, replace=False)
    sample_sims = embeddings[sample_idx] @ embeddings.T

    for name, hp, bids in methods:
        coloc_rates = []
        for i, si in enumerate(sample_idx):
            item_sims = sample_sims[i].copy()
            item_sims[si] = -np.inf
            nn_idx = np.argpartition(-item_sims, K)[:K]
            coloc = np.mean(bids[nn_idx] == bids[si])
            coloc_rates.append(coloc)
        print(f"  {name:>14}: {np.mean(coloc_rates):.4f}")
    print()

    # --- Recall curves ---
    print("Recall@10 by nprobe (margin-weighted routing):\n")
    print(f"{'nprobe':>7}", end="")
    for name, _, _ in methods:
        print(f" {name:>14}", end="")
    print(f" {'PCA lift':>10}")
    print("-" * (7 + 15 * len(methods) + 11))

    all_recalls = {}
    for name, hp, bids in methods:
        recalls = compute_recall_curve(
            embeddings, bids, hp, num_bits,
            query_indices, gt_all, nprobe_values)
        all_recalls[name] = recalls

    for np_val in nprobe_values:
        print(f"{np_val:>7}", end="")
        for name, _, _ in methods:
            print(f" {all_recalls[name][np_val]:>14.4f}", end="")
        # PCA lift over centroid PCA
        lift = all_recalls['RawPCA'][np_val] / max(all_recalls['CentroidPCA'][np_val], 1e-10)
        print(f" {lift:>10.2f}x")

    # --- Data fraction comparison ---
    print("\n\nRecall at matched data fraction:")
    print(f"{'ScanFrac':>9}", end="")
    for name, _, _ in methods:
        n_buckets = len(np.unique(methods[[n for n, _, _ in methods].index(name)][2]))
        print(f" {name:>14}", end="")
    print()

    for target_frac in [0.01, 0.02, 0.05, 0.10, 0.20]:
        print(f"{target_frac*100:>7.0f}%  ", end="")
        for name, hp, bids in methods:
            n_buckets = len(np.unique(bids))
            sizes = np.bincount(bids.astype(np.int64))
            sizes = sizes[sizes > 0]
            avg_sz = sizes.mean()
            nprobe = max(1, min(n_buckets, int(target_frac * N_ITEMS / avg_sz)))
            # Find nearest nprobe_value
            closest_np = min(nprobe_values, key=lambda x: abs(x - nprobe))
            recall = all_recalls[name].get(closest_np, 0)
            print(f" {recall:>10.4f}({nprobe:>2})", end="")
        print()

    # --- Multi-table with best hyperplane type ---
    print("\n\nMulti-table comparison (best single-table HP type):")
    print("Using Raw PCA hyperplanes with multiple independent tables\n")

    n_queries_mt = 500
    qi_mt = query_indices[:n_queries_mt]

    for n_tables in [1, 2, 3, 4]:
        for nprobe_per in [1, 2, 3]:
            # Build multiple PCA tables with different random rotations
            table_bids_list = []
            table_hp_list = []

            for t in range(n_tables):
                if t == 0:
                    hp_t = pca_hp  # First table uses straight PCA
                else:
                    # Perturbed PCA: add small random rotation
                    rng_t = np.random.default_rng(SEED + t * 100)
                    # Random orthogonal perturbation
                    noise = rng_t.standard_normal((num_bits, EMBEDDING_DIM)).astype(np.float32) * 0.3
                    hp_t = pca_hp + noise
                    # Re-orthogonalize via QR
                    Q, _ = np.linalg.qr(hp_t.T)
                    hp_t = Q[:, :num_bits].T.astype(np.float32)

                bids_t, _ = hash_all(embeddings, hp_t)
                table_bids_list.append(bids_t)
                table_hp_list.append(hp_t)

            recalls = []
            n_cands_list = []

            for qi in qi_mt:
                query = embeddings[qi]
                gt_set = gt_all[qi]
                candidate_set = set()

                for t in range(n_tables):
                    hp_t = table_hp_list[t]
                    bids_t = table_bids_list[t]
                    query_bid, projections = hash_query_with_proj(query, hp_t)
                    unique_bids = np.unique(bids_t)

                    bid_costs = []
                    for bid in unique_bids:
                        bid_int = int(bid)
                        if bid_int == query_bid:
                            bid_costs.append((0.0, bid_int))
                        else:
                            cost = margin_distance(query_bid, bid_int,
                                                   projections, num_bits)
                            bid_costs.append((cost, bid_int))
                    bid_costs.sort()

                    probe_bids = set(bid for _, bid in bid_costs[:nprobe_per])
                    mask = np.isin(bids_t, list(probe_bids))
                    candidate_set.update(np.where(mask)[0].tolist())

                found = len(gt_set & candidate_set)
                recalls.append(found / K)
                n_cands_list.append(len(candidate_set))

            mean_recall = np.mean(recalls)
            mean_cands = np.mean(n_cands_list)
            scan_frac = mean_cands / N_ITEMS

            label = f"PCA {num_bits}b x {n_tables}T x {nprobe_per}p"
            print(f"  {label:>25}: recall={mean_recall:.4f}, "
                  f"scan={scan_frac:.3f}, "
                  f"eff={mean_recall/max(scan_frac,1e-10):.2f}")


if __name__ == '__main__':
    main()
