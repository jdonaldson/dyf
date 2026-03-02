"""Experiment: hierarchical bridge routing vs flat LSH routing.

Key insight: bridge persistence is selective at coarse granularity (3 bits / 8
buckets, ~40% bridge rate) but universal at fine granularity (9 bits / 512
buckets, ~90% bridge rate). A hierarchical approach uses coarse bridge
connections to pick which macro-regions to explore, then fine margin routing
within those regions.

Methods compared:
1. Flat LSH (9 bits): margin-weighted multi-probe over a single flat partition
2. Hierarchical margin (3+6 bits): 2-level partition, margin routing at both
3. Hierarchical bridge (3+6 bits): coarse bridge-discounted routing + fine margin

All methods probe the same number of fine-level buckets (nprobe), so the
candidate set sizes are comparable.
"""

import numpy as np
import time
from collections import defaultdict


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
# Helpers
# ---------------------------------------------------------------------------

def hash_query(query, hyperplanes):
    """Compute LSH bucket ID and projections."""
    projections = hyperplanes @ query
    bits = (projections > 0).astype(np.uint64)
    bucket_id = np.uint64(0)
    for i, b in enumerate(bits):
        bucket_id |= (b << np.uint64(i))
    return int(bucket_id), projections


def margin_distance(a, b, projections, num_bits):
    """Sum of |projection[i]| for each differing bit."""
    xor = a ^ b
    cost = 0.0
    for i in range(num_bits):
        if xor & (1 << i):
            cost += abs(float(projections[i]))
    return cost


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_flat_index(embeddings, num_bits=9, seed=42):
    """Single-level LSH partition."""
    from dyf_rs import DensityClassifier

    n, dim = embeddings.shape
    clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits,
                            seed=seed, skip_isolation=True)
    clf.fit(embeddings)
    return {
        'bucket_ids': np.array(clf.get_bucket_ids()),
        'hyperplanes': np.array(clf.get_hyperplanes(), dtype=np.float32),
        'num_bits': num_bits,
    }


def build_hierarchical_index(embeddings, coarse_bits=3, fine_bits=6,
                              bridge_threshold=0.95, seed=42):
    """2-level index: coarse LSH (with bridge persistence) + fine LSH per bucket."""
    from dyf_rs import DensityClassifier

    n, dim = embeddings.shape

    # Level 1: Coarse partition
    coarse_clf = DensityClassifier(embedding_dim=dim, num_bits=coarse_bits,
                                   seed=seed, skip_isolation=True)
    coarse_clf.fit(embeddings)
    coarse_bids = np.array(coarse_clf.get_bucket_ids())
    coarse_hp = np.array(coarse_clf.get_hyperplanes(), dtype=np.float32)

    # Bridge persistence at coarse level
    bp = coarse_clf.bridge_persistence(embeddings, relative_threshold=bridge_threshold)
    pairs = bp.top_persistent_pairs(500)

    bridge_adj = defaultdict(set)
    for pair in pairs:
        bid_a, bid_b, persistence, _, _ = pair
        bridge_adj[bid_a].add(bid_b)
        bridge_adj[bid_b].add(bid_a)

    # Stats
    bridge_pers = np.array(bp.bridge_persistence)
    n_bridges = int(np.sum(bridge_pers > 0))

    # Level 2: Fine partition within each coarse bucket
    fine_data = {}
    unique_coarse = np.unique(coarse_bids)

    for cbid in unique_coarse:
        mask = coarse_bids == cbid
        indices = np.where(mask)[0]
        subset = embeddings[indices]

        if len(indices) < 20:
            fine_data[int(cbid)] = {
                'indices': indices,
                'bucket_ids': np.zeros(len(indices), dtype=np.int64),
                'hyperplanes': None,
                'num_bits': 0,
            }
            continue

        fine_clf = DensityClassifier(embedding_dim=dim, num_bits=fine_bits,
                                     seed=seed + int(cbid) + 1,
                                     skip_isolation=True)
        fine_clf.fit(subset)
        fine_bids = np.array(fine_clf.get_bucket_ids())
        fine_hp = np.array(fine_clf.get_hyperplanes(), dtype=np.float32)

        fine_data[int(cbid)] = {
            'indices': indices,
            'bucket_ids': fine_bids,
            'hyperplanes': fine_hp,
            'num_bits': fine_bits,
        }

    return {
        'coarse_bids': coarse_bids,
        'coarse_hp': coarse_hp,
        'coarse_bits': coarse_bits,
        'bridge_adj': dict(bridge_adj),
        'n_bridge_items': n_bridges,
        'n_bridge_pairs': len(pairs),
        'fine_data': fine_data,
    }


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def search_flat(query, embeddings, flat_index, nprobe=3):
    """Flat LSH search with margin-weighted multi-probe."""
    bucket_ids = flat_index['bucket_ids']
    hyperplanes = flat_index['hyperplanes']
    num_bits = flat_index['num_bits']

    query_bid, projections = hash_query(query, hyperplanes)

    unique_bids = np.unique(bucket_ids)
    bid_costs = []
    for bid in unique_bids:
        bid_int = int(bid)
        if bid_int == query_bid:
            bid_costs.append((0.0, bid_int))
        else:
            cost = margin_distance(query_bid, bid_int, projections, num_bits)
            bid_costs.append((cost, bid_int))

    bid_costs.sort()
    probe_bids = [bid for _, bid in bid_costs[:nprobe]]

    mask = np.isin(bucket_ids, probe_bids)
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

    sims = embeddings[candidate_idx] @ query
    topk_local = np.argsort(-sims)[:K]
    return candidate_idx[topk_local], sims[topk_local], len(candidate_idx)


def search_hierarchical(query, embeddings, hier_index, nprobe=3,
                         use_bridges=True, bridge_discount=0.5):
    """Hierarchical search with global priority queue.

    Builds a priority queue of (total_cost, coarse_bid, fine_bid) where
    total_cost = coarse_margin_cost + fine_margin_cost. Bridge routing
    discounts coarse_cost for bridge-adjacent buckets.

    Probes the top-nprobe fine buckets from this global ranking.
    """
    coarse_hp = hier_index['coarse_hp']
    coarse_bits = hier_index['coarse_bits']
    bridge_adj = hier_index['bridge_adj']
    fine_data = hier_index['fine_data']

    # Coarse routing
    query_cbid, coarse_proj = hash_query(query, coarse_hp)

    # Build global priority queue over all (coarse, fine) bucket pairs
    pq = []
    for cbid in sorted(fine_data.keys()):
        # Coarse cost
        if cbid == query_cbid:
            coarse_cost = 0.0
        else:
            coarse_cost = margin_distance(query_cbid, cbid, coarse_proj,
                                          coarse_bits)
            if use_bridges and cbid in bridge_adj.get(query_cbid, set()):
                coarse_cost *= (1.0 - bridge_discount)

        fd = fine_data[cbid]
        fine_hp = fd['hyperplanes']
        fine_bits = fd['num_bits']
        fine_bids = fd['bucket_ids']

        if fine_hp is None or fine_bits == 0:
            # No fine partition — single "bucket" containing all items
            pq.append((coarse_cost, cbid, -1))
        else:
            fine_bid, fine_proj = hash_query(query, fine_hp)
            unique_fine = np.unique(fine_bids)
            for fbid in unique_fine:
                fbid_int = int(fbid)
                if fbid_int == fine_bid:
                    fine_cost = 0.0
                else:
                    fine_cost = margin_distance(fine_bid, fbid_int,
                                                fine_proj, fine_bits)
                pq.append((coarse_cost + fine_cost, cbid, fbid_int))

    pq.sort()

    # Probe top-nprobe entries
    candidate_idx = []
    for cost, cbid, fbid in pq[:nprobe]:
        fd = fine_data[cbid]
        if fbid == -1:
            candidate_idx.extend(fd['indices'].tolist())
        else:
            mask = fd['bucket_ids'] == fbid
            candidate_idx.extend(fd['indices'][mask].tolist())

    if not candidate_idx:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

    candidate_idx = np.unique(np.array(candidate_idx, dtype=np.int64))
    sims = embeddings[candidate_idx] @ query
    topk_local = np.argsort(-sims)[:K]
    return candidate_idx[topk_local], sims[topk_local], len(candidate_idx)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cross_cluster_recall(gt_indices, result_indices, assignments, query_cluster):
    """What fraction of true cross-cluster results were found?"""
    gt_set = set(int(x) for x in gt_indices)
    res_set = set(int(x) for x in result_indices)
    gt_cross = {i for i in gt_set if assignments[i] != query_cluster}
    if not gt_cross:
        return float('nan')
    return len(gt_cross & res_set) / len(gt_cross)


def result_diversity(result_indices, assignments):
    """Number of distinct clusters in the result set."""
    if len(result_indices) == 0:
        return 0
    return len(set(int(assignments[i]) for i in result_indices))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=== Hierarchical Bridge Routing Experiment ===\n")

    print("Generating clustered embeddings...")
    embeddings, assignments = generate_clustered_data(
        N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)
    print(f"  {N_ITEMS} items, {EMBEDDING_DIM}d, {N_CLUSTERS} clusters\n")

    # Build indices
    print("Building flat index (9 bits)...")
    t0 = time.perf_counter()
    flat_idx = build_flat_index(embeddings, num_bits=9, seed=SEED)
    n_flat_buckets = len(np.unique(flat_idx['bucket_ids']))
    print(f"  {n_flat_buckets} buckets, {time.perf_counter()-t0:.2f}s\n")

    print("Building hierarchical index (3 coarse + 6 fine bits)...")
    t0 = time.perf_counter()
    hier_idx = build_hierarchical_index(
        embeddings, coarse_bits=3, fine_bits=6,
        bridge_threshold=0.95, seed=SEED)
    hier_time = time.perf_counter() - t0
    n_coarse = len(np.unique(hier_idx['coarse_bids']))
    n_fine_total = sum(
        len(np.unique(fd['bucket_ids'])) for fd in hier_idx['fine_data'].values()
    )
    print(f"  {n_coarse} coarse buckets, {n_fine_total} total fine buckets")
    print(f"  Bridge items: {hier_idx['n_bridge_items']} / {N_ITEMS} "
          f"({100*hier_idx['n_bridge_items']/N_ITEMS:.1f}%)")
    print(f"  Bridge-connected coarse pairs: {hier_idx['n_bridge_pairs']}")

    # Show bridge adjacency map
    print("  Coarse bridge adjacency:")
    for cbid in sorted(hier_idx['bridge_adj'].keys()):
        adj = sorted(hier_idx['bridge_adj'][cbid])
        print(f"    bucket {cbid} -> {adj}")
    print(f"  Build time: {hier_time:.2f}s\n")

    # Sample queries
    rng = np.random.default_rng(SEED + 1)
    n_queries = 500
    query_indices = rng.choice(N_ITEMS, size=n_queries, replace=False)

    methods = ['flat_9bit', 'hier_margin', 'hier_bridge']

    for nprobe in [1, 3, 5, 10, 20]:
        metrics = {m: {'recall': [], 'cross_recall': [], 'diversity': [],
                       'n_cands': []}
                   for m in methods}

        for qi in query_indices:
            query = embeddings[qi]
            query_cluster = assignments[qi]
            gt_idx, _ = brute_force_topk(embeddings, query, K)
            gt_set = set(int(x) for x in gt_idx)

            # Flat LSH (9 bits)
            flat_res, _, flat_ncands = search_flat(
                query, embeddings, flat_idx, nprobe=nprobe)

            # Hierarchical margin (no bridges)
            hm_res, _, hm_ncands = search_hierarchical(
                query, embeddings, hier_idx, nprobe=nprobe,
                use_bridges=False)

            # Hierarchical bridge
            hb_res, _, hb_ncands = search_hierarchical(
                query, embeddings, hier_idx, nprobe=nprobe,
                use_bridges=True, bridge_discount=0.5)

            for name, res_idx, ncands in [
                ('flat_9bit', flat_res, flat_ncands),
                ('hier_margin', hm_res, hm_ncands),
                ('hier_bridge', hb_res, hb_ncands),
            ]:
                res_set = set(int(x) for x in res_idx)
                recall = len(gt_set & res_set) / K
                cr = cross_cluster_recall(gt_idx, res_idx, assignments,
                                          query_cluster)
                div = result_diversity(res_idx, assignments)
                metrics[name]['recall'].append(recall)
                if not np.isnan(cr):
                    metrics[name]['cross_recall'].append(cr)
                metrics[name]['diversity'].append(div)
                metrics[name]['n_cands'].append(ncands)

        print(f"--- nprobe={nprobe} ---")
        print(f"{'Method':<15} {'Recall@10':>10} {'CrossRecall':>12} "
              f"{'Diversity':>10} {'AvgCands':>10}")
        for name in methods:
            m = metrics[name]
            print(f"{name:<15} {np.mean(m['recall']):>10.4f} "
                  f"{np.mean(m['cross_recall']):>12.4f} "
                  f"{np.mean(m['diversity']):>10.2f} "
                  f"{np.mean(m['n_cands']):>10.0f}")

        # Show bridge advantage/disadvantage
        bridge_recall = np.mean(metrics['hier_bridge']['recall'])
        margin_recall = np.mean(metrics['hier_margin']['recall'])
        flat_recall = np.mean(metrics['flat_9bit']['recall'])
        bridge_cr = np.mean(metrics['hier_bridge']['cross_recall'])
        margin_cr = np.mean(metrics['hier_margin']['cross_recall'])
        print(f"  Bridge vs margin: recall {bridge_recall-margin_recall:+.4f}, "
              f"cross-recall {bridge_cr-margin_cr:+.4f}")
        print(f"  Bridge vs flat:   recall {bridge_recall-flat_recall:+.4f}")
        print()


if __name__ == '__main__':
    main()
