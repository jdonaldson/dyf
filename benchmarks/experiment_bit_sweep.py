"""Bit-level sweep: at which LSH granularity does tree routing help most?

For each num_bits (1-12), measure:
- Recall ceiling at various nprobe values
- Data fraction scanned at each nprobe (nprobe * avg_bucket_size / N)
- Recall per data fraction = routing efficiency

Also tests multi-table LSH (M independent hash tables, union of candidates)
to see how stability hashing improves coverage.
"""

import numpy as np
import time
from collections import Counter


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


def hash_query(query, hyperplanes):
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


def margin_rank_all_gt(query, gt_idx, bucket_ids, hyperplanes, num_bits):
    """For each GT item, return its bucket's margin rank."""
    query_bid, projections = hash_query(query, hyperplanes)
    unique_bids = np.unique(bucket_ids)

    # Rank all buckets by margin distance
    bid_costs = []
    for bid in unique_bids:
        bid_int = int(bid)
        if bid_int == query_bid:
            bid_costs.append((0.0, bid_int))
        else:
            cost = margin_distance(query_bid, bid_int, projections, num_bits)
            bid_costs.append((cost, bid_int))
    bid_costs.sort()
    bid_to_rank = {bid: rank for rank, (_, bid) in enumerate(bid_costs)}

    ranks = []
    for gi in gt_idx:
        gt_bid = int(bucket_ids[gi])
        ranks.append(bid_to_rank.get(gt_bid, len(unique_bids)))
    return ranks


# ---------------------------------------------------------------------------
# Part 1: Single-table bit sweep
# ---------------------------------------------------------------------------

def run_bit_sweep(embeddings):
    from dyf_rs import DensityClassifier

    print("=" * 70)
    print("  PART 1: Single-table bit sweep")
    print("=" * 70)

    rng = np.random.default_rng(SEED + 1)
    n_queries = 1000
    query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

    # Pre-compute ground truth
    gt_all = {}
    for qi in query_indices:
        gt_idx, _ = brute_force_topk(embeddings, embeddings[qi], K)
        gt_all[qi] = gt_idx

    # Target data fractions to compare across bit levels
    target_fractions = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

    results = {}  # num_bits -> {fraction -> recall}

    for num_bits in range(1, 13):
        dim = embeddings.shape[1]
        clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits,
                                seed=SEED, skip_isolation=True)
        clf.fit(embeddings)
        bucket_ids = clf.get_bucket_ids()
        hyperplanes = clf.get_hyperplanes()

        n_buckets = len(np.unique(bucket_ids))
        bucket_sizes = Counter(bucket_ids.tolist())
        avg_bucket_size = np.mean(list(bucket_sizes.values()))

        # For each query, get margin ranks of all GT items
        all_gt_ranks = []
        for qi in query_indices:
            ranks = margin_rank_all_gt(embeddings[qi], gt_all[qi],
                                       bucket_ids, hyperplanes, num_bits)
            all_gt_ranks.extend(ranks)

        all_gt_ranks = np.array(all_gt_ranks)

        # Compute recall at each nprobe
        row = {'n_buckets': n_buckets, 'avg_bucket': avg_bucket_size}
        for frac in target_fractions:
            # nprobe that gives approximately this data fraction
            nprobe = max(1, int(frac * N_ITEMS / avg_bucket_size))
            nprobe = min(nprobe, n_buckets)
            actual_frac = nprobe * avg_bucket_size / N_ITEMS
            recall = np.mean(all_gt_ranks < nprobe)
            row[frac] = (nprobe, actual_frac, recall)

        results[num_bits] = row

    # Print table
    print(f"\n{'bits':>4} {'buckets':>8} {'avg_sz':>7}", end="")
    for frac in target_fractions:
        print(f" | {frac*100:>4.1f}% scan", end="")
    print()
    print("-" * (25 + 14 * len(target_fractions)))

    for num_bits in range(1, 13):
        r = results[num_bits]
        print(f"{num_bits:>4} {r['n_buckets']:>8} {r['avg_bucket']:>7.0f}", end="")
        for frac in target_fractions:
            nprobe, actual_frac, recall = r[frac]
            print(f" | {recall:>5.3f} ({nprobe:>3})", end="")
        print()

    # Find best bits for each target fraction
    print(f"\nBest bits per scan budget:")
    for frac in target_fractions:
        best_bits = max(range(1, 13),
                        key=lambda b: results[b][frac][2])
        nprobe, actual_frac, recall = results[best_bits][frac]
        print(f"  {frac*100:>4.1f}% scan: {best_bits} bits "
              f"(recall={recall:.3f}, nprobe={nprobe})")

    return results


# ---------------------------------------------------------------------------
# Part 2: Multi-table LSH
# ---------------------------------------------------------------------------

def run_multi_table(embeddings):
    from dyf_rs import DensityClassifier

    print("\n" + "=" * 70)
    print("  PART 2: Multi-table LSH (union of M independent hash tables)")
    print("=" * 70)

    rng = np.random.default_rng(SEED + 1)
    n_queries = 500
    query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

    gt_all = {}
    for qi in query_indices:
        gt_idx, _ = brute_force_topk(embeddings, embeddings[qi], K)
        gt_all[qi] = set(int(x) for x in gt_idx)

    dim = embeddings.shape[1]

    # Test configurations: (num_bits, num_tables, nprobe_per_table)
    configs = []
    # Single table baselines
    for bits in [3, 4, 5, 6, 9]:
        configs.append((bits, 1, 3))
        configs.append((bits, 1, 5))
        configs.append((bits, 1, 10))

    # Multi-table: fewer bits but multiple tables
    for bits in [2, 3, 4, 5]:
        for n_tables in [2, 3, 4, 5]:
            configs.append((bits, n_tables, 1))  # primary only per table
            configs.append((bits, n_tables, 2))  # primary + 1 alt per table

    print(f"\n{'Config':>20} {'Recall@10':>10} {'AvgCands':>10} "
          f"{'ScanFrac':>10} {'Recall/Scan':>12}")
    print("-" * 65)

    seen = set()
    results = []

    for bits, n_tables, nprobe_per in configs:
        key = (bits, n_tables, nprobe_per)
        if key in seen:
            continue
        seen.add(key)

        # Build tables
        tables = []
        for t in range(n_tables):
            clf = DensityClassifier(embedding_dim=dim, num_bits=bits,
                                    seed=SEED + t * 100, skip_isolation=True)
            clf.fit(embeddings)
            tables.append({
                'bucket_ids': clf.get_bucket_ids(),
                'hyperplanes': clf.get_hyperplanes(),
            })

        recalls = []
        n_cands_list = []

        for qi in query_indices:
            query = embeddings[qi]
            gt_set = gt_all[qi]

            # Collect candidates from all tables
            candidate_set = set()
            for table in tables:
                bids = table['bucket_ids']
                hp = table['hyperplanes']
                query_bid, projections = hash_query(query, hp)
                unique_bids = np.unique(bids)

                # Rank buckets by margin
                bid_costs = []
                for bid in unique_bids:
                    bid_int = int(bid)
                    if bid_int == query_bid:
                        bid_costs.append((0.0, bid_int))
                    else:
                        cost = margin_distance(query_bid, bid_int,
                                               projections, bits)
                        bid_costs.append((cost, bid_int))
                bid_costs.sort()

                probe_bids = set(bid for _, bid in bid_costs[:nprobe_per])
                mask = np.isin(bids, list(probe_bids))
                candidate_set.update(np.where(mask)[0].tolist())

            n_cands = len(candidate_set)
            found = len(gt_set & candidate_set)
            recalls.append(found / K)
            n_cands_list.append(n_cands)

        mean_recall = np.mean(recalls)
        mean_cands = np.mean(n_cands_list)
        scan_frac = mean_cands / N_ITEMS
        efficiency = mean_recall / max(scan_frac, 1e-10)

        label = f"{bits}b x {n_tables}T x {nprobe_per}p"
        results.append((label, mean_recall, mean_cands, scan_frac, efficiency))

    # Sort by efficiency
    results.sort(key=lambda x: -x[4])

    for label, recall, cands, scan, eff in results:
        print(f"{label:>20} {recall:>10.4f} {cands:>10.0f} "
              f"{scan:>10.3f} {eff:>12.2f}")

    # Show top-10 by efficiency
    print(f"\nTop 10 by recall/scan efficiency:")
    for i, (label, recall, cands, scan, eff) in enumerate(results[:10]):
        print(f"  {i+1}. {label}: recall={recall:.4f}, "
              f"scan={scan:.3f}, eff={eff:.2f}")

    # Show top-10 by raw recall
    by_recall = sorted(results, key=lambda x: -x[1])
    print(f"\nTop 10 by raw recall:")
    for i, (label, recall, cands, scan, eff) in enumerate(by_recall[:10]):
        print(f"  {i+1}. {label}: recall={recall:.4f}, "
              f"scan={scan:.3f}, eff={eff:.2f}")


# ---------------------------------------------------------------------------
# Part 3: Hierarchical bit analysis (tree depth effect)
# ---------------------------------------------------------------------------

def run_tree_depth_analysis(embeddings):
    from dyf_rs import DensityClassifier

    print("\n" + "=" * 70)
    print("  PART 3: Hierarchical routing — tree depth effect")
    print("=" * 70)
    print("  Simulates tree traversal: at each depth level, route with b bits,")
    print("  then recurse into selected children.\n")

    rng = np.random.default_rng(SEED + 1)
    n_queries = 500
    query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

    gt_all = {}
    for qi in query_indices:
        gt_idx, _ = brute_force_topk(embeddings, embeddings[qi], K)
        gt_all[qi] = set(int(x) for x in gt_idx)

    dim = embeddings.shape[1]

    # For each (bits_per_level, depth), build a simulated tree and measure
    # recall of GT items that are reachable via primary-path traversal
    for bits in [2, 3, 4]:
        print(f"\n--- {bits} bits per level ---")
        print(f"{'Depth':>5} {'TotalBits':>10} {'EffBuckets':>11} "
              f"{'PrimaryRcl':>11} {'AvgLeafSz':>10}")

        for depth in range(1, 6):
            total_bits = bits * depth

            # Build tree: at each level, partition the items that reached
            # this node. Track which items are reachable via primary path.
            # Level 0: all items
            # Level 1: primary bucket of level-0 classifier
            # Level 2: primary bucket of level-1 classifier (within level-1 items)
            # etc.

            primary_recalls = []
            leaf_sizes = []

            for qi in query_indices:
                query = embeddings[qi]
                gt_set = gt_all[qi]

                # Simulate primary-path traversal
                current_indices = np.arange(N_ITEMS)

                for d in range(depth):
                    if len(current_indices) < 4:
                        break

                    subset = embeddings[current_indices]
                    clf = DensityClassifier(
                        embedding_dim=dim, num_bits=bits,
                        seed=SEED + d * 17 + qi % 100,  # vary by depth & query batch
                        skip_isolation=True)
                    clf.fit(subset)
                    sub_bids = clf.get_bucket_ids()
                    sub_hp = clf.get_hyperplanes()

                    # Route query
                    query_bid, _ = hash_query(query, sub_hp)

                    # Find items in primary bucket
                    mask = sub_bids == query_bid
                    if not np.any(mask):
                        # Fallback: nearest bucket
                        unique = np.unique(sub_bids)
                        projections = sub_hp @ query
                        best_bid = min(unique,
                                       key=lambda b: margin_distance(
                                           query_bid, int(b), projections, bits))
                        mask = sub_bids == best_bid

                    current_indices = current_indices[mask]

                leaf_sizes.append(len(current_indices))
                # How many GT items made it to the leaf?
                leaf_set = set(current_indices.tolist())
                found = len(gt_set & leaf_set)
                primary_recalls.append(found / K)

            mean_recall = np.mean(primary_recalls)
            mean_leaf = np.mean(leaf_sizes)
            eff_buckets = N_ITEMS / max(mean_leaf, 1)

            print(f"{depth:>5} {total_bits:>10} {eff_buckets:>11.0f} "
                  f"{mean_recall:>11.4f} {mean_leaf:>10.0f}")


def main():
    print("=== Bit-Level Sweep & Multi-Table Analysis ===\n")

    embeddings, assignments = generate_clustered_data(
        N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)
    print(f"{N_ITEMS} items, {EMBEDDING_DIM}d, {N_CLUSTERS} clusters\n")

    run_bit_sweep(embeddings)
    run_multi_table(embeddings)
    run_tree_depth_analysis(embeddings)


if __name__ == '__main__':
    main()
