"""Backward analysis: where do ground-truth results land in the LSH structure?

Instead of forward search (route → find), we work backwards:
1. For each query, find brute-force top-k ground truth
2. For each GT result, determine which bucket it landed in
3. Measure: Hamming distance, margin distance, bridge status, rank of that
   bucket in our routing order
4. Diagnose exactly WHY results are missed and what signal could find them

This tells us the theoretical upper bound of each routing strategy and
where the gap is.
"""

import numpy as np
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LSH helpers
# ---------------------------------------------------------------------------

def hash_query(query, hyperplanes):
    projections = hyperplanes @ query
    bits = (projections > 0).astype(np.uint64)
    bucket_id = np.uint64(0)
    for i, b in enumerate(bits):
        bucket_id |= (b << np.uint64(i))
    return int(bucket_id), projections


def hamming_distance(a, b, num_bits):
    xor = a ^ b
    return bin(xor & ((1 << num_bits) - 1)).count('1')


def margin_distance(a, b, projections, num_bits):
    xor = a ^ b
    cost = 0.0
    for i in range(num_bits):
        if xor & (1 << i):
            cost += abs(float(projections[i]))
    return cost


def bucket_margin_rank(query_bid, target_bid, all_bids, projections, num_bits):
    """What rank does target_bid get in margin-ordered probe list?"""
    unique_bids = np.unique(all_bids)
    costs = []
    for bid in unique_bids:
        bid_int = int(bid)
        if bid_int == query_bid:
            costs.append((0.0, bid_int))
        else:
            cost = margin_distance(query_bid, bid_int, projections, num_bits)
            costs.append((cost, bid_int))
    costs.sort()
    for rank, (_, bid) in enumerate(costs):
        if bid == target_bid:
            return rank
    return len(costs)  # not found


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=== Backward Analysis: Where Do GT Results Land? ===\n")

    embeddings, assignments = generate_clustered_data(
        N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)
    print(f"{N_ITEMS} items, {EMBEDDING_DIM}d, {N_CLUSTERS} clusters\n")

    from dyf_rs import DensityClassifier

    for num_bits in [3, 6, 9]:
        print(f"{'='*60}")
        print(f"  Analyzing {num_bits}-bit LSH partition")
        print(f"{'='*60}\n")

        clf = DensityClassifier(embedding_dim=EMBEDDING_DIM, num_bits=num_bits,
                                seed=SEED, skip_isolation=True)
        clf.fit(embeddings)
        bucket_ids = clf.get_bucket_ids()
        hyperplanes = clf.get_hyperplanes()
        n_buckets = len(np.unique(bucket_ids))

        # Bridge data
        bp = clf.bridge_persistence(embeddings, relative_threshold=0.95)
        bridge_pers = np.array(bp.bridge_persistence)
        bridge_ratio = np.array(bp.bridge_ratio)

        # Bucket size distribution
        bucket_sizes = Counter(bucket_ids.tolist())
        sizes = list(bucket_sizes.values())
        print(f"Buckets: {n_buckets} (of {2**num_bits} possible)")
        print(f"Bucket sizes: min={min(sizes)}, median={np.median(sizes):.0f}, "
              f"max={max(sizes)}, mean={np.mean(sizes):.0f}")
        print(f"Bridge items: {np.sum(bridge_pers > 0)} / {N_ITEMS} "
              f"({100*np.sum(bridge_pers > 0)/N_ITEMS:.1f}%)\n")

        # Sample queries
        rng = np.random.default_rng(SEED + 1)
        n_queries = 1000
        query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

        # Per-GT-result analysis
        gt_hamming_dists = []       # Hamming dist from query bucket to GT bucket
        gt_margin_ranks = []        # Margin-order rank of GT bucket
        gt_in_primary = 0           # GT in query's own bucket
        gt_is_bridge = 0            # GT item is a bridge item
        gt_same_cluster = 0         # GT item in same cluster as query
        gt_bridge_ratio_vals = []   # Bridge ratio of GT items that are bridges
        gt_total = 0

        # Per-query analysis
        query_primary_recall = []   # How many GT in primary bucket
        query_min_nprobe = []       # Min nprobe to get ALL GT items

        # Detailed: which bits differ for GT items not in primary bucket
        bit_flip_counts = np.zeros(num_bits, dtype=int)

        # Track: for GT items NOT in primary, what's their margin for
        # the flipped bit(s)?
        flip_margins = []  # (bit_index, |projection|) for each flipped bit

        for qi in query_indices:
            query = embeddings[qi]
            query_cluster = assignments[qi]
            query_bid, projections = hash_query(query, hyperplanes)

            gt_idx, gt_scores = brute_force_topk(embeddings, query, K)

            n_in_primary = 0
            max_rank_needed = 0

            for gt_i, gt_score in zip(gt_idx, gt_scores):
                gt_bid = int(bucket_ids[gt_i])
                gt_total += 1

                # Hamming distance
                hdist = hamming_distance(query_bid, gt_bid, num_bits)
                gt_hamming_dists.append(hdist)

                # Margin rank
                mrank = bucket_margin_rank(query_bid, gt_bid,
                                           bucket_ids, projections, num_bits)
                gt_margin_ranks.append(mrank)
                max_rank_needed = max(max_rank_needed, mrank)

                # Primary bucket?
                if gt_bid == query_bid:
                    gt_in_primary += 1
                    n_in_primary += 1
                else:
                    # Which bits differ?
                    xor = query_bid ^ gt_bid
                    for i in range(num_bits):
                        if xor & (1 << i):
                            bit_flip_counts[i] += 1
                            flip_margins.append((i, abs(float(projections[i]))))

                # Bridge status
                if bridge_pers[gt_i] > 0:
                    gt_is_bridge += 1
                    gt_bridge_ratio_vals.append(float(bridge_ratio[gt_i]))

                # Same cluster?
                if assignments[gt_i] == query_cluster:
                    gt_same_cluster += 1

            query_primary_recall.append(n_in_primary / K)
            query_min_nprobe.append(max_rank_needed + 1)

        # Report
        print("--- Where GT results land ---")
        print(f"GT items in query's primary bucket: {gt_in_primary}/{gt_total} "
              f"({100*gt_in_primary/gt_total:.1f}%)")
        print(f"GT items in same cluster as query: {gt_same_cluster}/{gt_total} "
              f"({100*gt_same_cluster/gt_total:.1f}%)")
        print(f"GT items that are bridge items: {gt_is_bridge}/{gt_total} "
              f"({100*gt_is_bridge/gt_total:.1f}%)")
        if gt_bridge_ratio_vals:
            print(f"  Bridge ratio (GT bridges): mean={np.mean(gt_bridge_ratio_vals):.3f}, "
                  f"median={np.median(gt_bridge_ratio_vals):.3f}")
        print()

        print("--- Hamming distance distribution (query bucket → GT bucket) ---")
        hd_counter = Counter(gt_hamming_dists)
        for d in sorted(hd_counter.keys()):
            pct = 100 * hd_counter[d] / gt_total
            bar = '#' * int(pct / 2)
            print(f"  Hamming {d}: {hd_counter[d]:>5} ({pct:>5.1f}%) {bar}")
        print()

        print("--- Margin rank of GT bucket (0=primary, 1=best alternative, ...) ---")
        rank_arr = np.array(gt_margin_ranks)
        for pct in [50, 75, 90, 95, 99]:
            print(f"  p{pct}: rank {np.percentile(rank_arr, pct):.0f}")
        print(f"  max: rank {rank_arr.max()}")
        print()

        # Cumulative recall by nprobe
        print("--- Cumulative recall@10 by nprobe (margin routing) ---")
        for nprobe_target in [1, 2, 3, 5, 10, 20, 50]:
            if nprobe_target > n_buckets:
                break
            # What fraction of GT items are in the top-nprobe buckets?
            in_range = sum(1 for r in gt_margin_ranks if r < nprobe_target)
            recall = in_range / gt_total
            print(f"  nprobe={nprobe_target:>3}: recall={recall:.4f} "
                  f"({in_range}/{gt_total} GT items reachable)")
        print()

        # Which bits get flipped most for missed GT items
        if num_bits <= 12:
            print("--- Bit flip frequency for GT items NOT in primary bucket ---")
            total_not_primary = gt_total - gt_in_primary
            for i in range(num_bits):
                pct = 100 * bit_flip_counts[i] / max(total_not_primary, 1)
                # Also show the mean margin for this bit
                bit_margins = [m for bi, m in flip_margins if bi == i]
                mean_margin = np.mean(bit_margins) if bit_margins else 0
                print(f"  bit {i}: flipped {bit_flip_counts[i]:>5} times "
                      f"({pct:>5.1f}%), mean |projection|={mean_margin:.4f}")
            print()

        # Per-query: how many queries have ALL GT in primary?
        all_in_primary = sum(1 for r in query_primary_recall if r == 1.0)
        print(f"Queries with all GT in primary: {all_in_primary}/{n_queries} "
              f"({100*all_in_primary/n_queries:.1f}%)")
        print(f"Mean primary-only recall: {np.mean(query_primary_recall):.4f}")

        # Min nprobe to reach all GT items
        mnp = np.array(query_min_nprobe)
        print(f"Min nprobe for perfect recall: "
              f"median={np.median(mnp):.0f}, p90={np.percentile(mnp, 90):.0f}, "
              f"p99={np.percentile(mnp, 99):.0f}, max={mnp.max()}")
        print()

        # Detailed look at a few queries where GT is far from primary
        print("--- Example queries where GT items are far from primary ---")
        worst_queries = np.argsort([-m for m in query_min_nprobe])[:3]
        for wq_pos in worst_queries:
            qi = query_indices[wq_pos]
            query = embeddings[qi]
            query_bid, projections = hash_query(query, hyperplanes)
            gt_idx, gt_scores = brute_force_topk(embeddings, query, K)

            print(f"\n  Query {qi} (cluster {assignments[qi]}, bucket {query_bid}):")
            print(f"  {'Rank':>4} {'Item':>7} {'Score':>7} {'Bucket':>7} "
                  f"{'Hamm':>5} {'MRank':>6} {'Bridge':>7} {'Cluster':>8}")
            for rank, (gt_i, gt_s) in enumerate(zip(gt_idx, gt_scores)):
                gt_bid = int(bucket_ids[gt_i])
                hdist = hamming_distance(query_bid, gt_bid, num_bits)
                mrank = bucket_margin_rank(query_bid, gt_bid,
                                           bucket_ids, projections, num_bits)
                is_bridge = "Y" if bridge_pers[gt_i] > 0 else "N"
                cl = assignments[gt_i]
                same = "*" if cl == assignments[qi] else " "
                print(f"  {rank+1:>4} {gt_i:>7} {gt_s:>7.4f} {gt_bid:>7} "
                      f"{hdist:>5} {mrank:>6} {is_bridge:>7} {cl:>4}{same}")

        print("\n")


if __name__ == '__main__':
    main()
