"""
HNSW Promotion Experiment: Multi-Address vs Random

HNSW promotes points to upper layers randomly. The hypothesis: multi-address
bridge points — identified for free during PCA tree construction — are better
upper-layer candidates because they connect distinct semantic regions.

This script builds a minimal 2-layer navigable small-world graph and compares
search recall under three promotion strategies:
  1. Random (standard HNSW)
  2. Multi-address (depth-weighted PCA tree boundary score)
  3. Simple LSH (low centroid similarity from DensityClassifier)

If multi-address promotion achieves higher recall at the same compute budget,
it can replace HNSW's random promotion as a drop-in improvement.

Usage:
    python demo/hnsw_promotion.py demo/wiki_simple_50k.parquet --sample 10000
"""

import argparse
import time

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from dyf.pca_tree import build_pca_tree, extract_boundary_persistence


def build_layer0_graphs(embeddings, k=32, bridge_indices=None):
    """Build layer 0 in five variants for comparison.

    Returns dict of {name: graph_dict}, dict of {name: edge_count}.

    Variants:
    - "directed": A→B if B is in A's kNN (asymmetric)
    - "bidirectional": A↔B if either is in the other's kNN (unpruned)
    - "bridge-bidir": directed + symmetric edges at bridge points (unpruned)
    - "pruned-bidir": bidirectional, then cap each point at k best neighbors
    - "pruned-bridge": bridge-bidir, then cap each point at k best neighbors
    """
    n = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Directed: just the forward kNN edges
    directed_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in indices[i, 1:]:
            directed_sets[i].add(int(j))

    # Full bidirectional: symmetrize everything
    bidir_sets = [set(s) for s in directed_sets]
    for i in range(n):
        for j in directed_sets[i]:
            bidir_sets[j].add(i)

    # Bridge-only bidirectional: symmetrize only at bridges
    bridge_set = set(bridge_indices.tolist()) if bridge_indices is not None else set()
    bridge_bidir_sets = [set(s) for s in directed_sets]
    for i in range(n):
        for j in directed_sets[i]:
            if i in bridge_set or j in bridge_set:
                bridge_bidir_sets[j].add(i)

    def prune(neighbor_sets, max_k):
        """Keep only the max_k highest-similarity neighbors per point."""
        pruned = [None] * n
        for i in range(n):
            nbs = list(neighbor_sets[i])
            if len(nbs) <= max_k:
                pruned[i] = set(nbs)
            else:
                nbs_arr = np.array(nbs, dtype=np.int32)
                sims = emb_normed[nbs_arr] @ emb_normed[i]
                top_k_idx = np.argpartition(-sims, max_k)[:max_k]
                pruned[i] = set(nbs_arr[top_k_idx].tolist())
        return pruned

    pruned_bidir_sets = prune(bidir_sets, k)
    pruned_bridge_sets = prune(bridge_bidir_sets, k)

    def to_dict(sets):
        return {i: np.array(sorted(s), dtype=np.int32) for i, s in enumerate(sets)}

    variants = {
        "directed":      directed_sets,
        "bidirectional":  bidir_sets,
        "bridge-bidir":  bridge_bidir_sets,
        "pruned-bidir":  pruned_bidir_sets,
        "pruned-bridge": pruned_bridge_sets,
    }

    graphs = {}
    counts = {}
    for name, sets in variants.items():
        graphs[name] = to_dict(sets)
        counts[name] = sum(len(s) for s in sets)

    return graphs, counts


def build_upper_layer_graph(embeddings_normed, promoted_indices, k=16):
    """Build bidirectional kNN graph among promoted points only."""
    m = len(promoted_indices)
    sub_emb = embeddings_normed[promoted_indices]
    sims = sub_emb @ sub_emb.T

    k_actual = min(k, m - 1)
    neighbor_sets = {int(promoted_indices[i]): set() for i in range(m)}

    for i in range(m):
        row_sims = sims[i].copy()
        row_sims[i] = -2  # exclude self
        if k_actual > 0:
            top_k = np.argpartition(-row_sims, k_actual)[:k_actual]
            gi = int(promoted_indices[i])
            for j in top_k:
                gj = int(promoted_indices[j])
                neighbor_sets[gi].add(gj)
                neighbor_sets[gj].add(gi)  # symmetrize

    neighbors = {}
    for gi, nbs in neighbor_sets.items():
        neighbors[gi] = np.array(sorted(nbs), dtype=np.int32)

    return neighbors


def _get_neighbors(graph, node):
    """Get neighbors from either a dict or list graph representation."""
    if isinstance(graph, dict):
        return graph.get(node, np.array([], dtype=np.int32))
    return graph[node] if node < len(graph) else np.array([], dtype=np.int32)


def greedy_search(query_vec, entry_point, graph_neighbors, embeddings_normed,
                  ef=1, exclude=-1):
    """Greedy beam search on a graph layer.

    Returns (best_indices, n_distance_computations).
    best_indices are sorted by similarity (descending).
    """
    import heapq

    visited = set()
    sim = float(embeddings_normed[entry_point] @ query_vec)
    candidates = [(-sim, entry_point)]  # min-heap by negative sim
    visited.add(entry_point)
    n_dist = 1

    results = [(-sim, entry_point)]

    while candidates:
        neg_sim, current = heapq.heappop(candidates)

        # If current is worse than worst in results and results is full, stop
        if len(results) >= ef and -neg_sim < -results[0][0]:
            break

        for nb in _get_neighbors(graph_neighbors, current):
            nb = int(nb)
            if nb in visited or nb == exclude:
                continue
            visited.add(nb)
            s = float(embeddings_normed[nb] @ query_vec)
            n_dist += 1

            if len(results) < ef or s > -results[0][0]:
                heapq.heappush(candidates, (-s, nb))
                heapq.heappush(results, (-s, nb))
                if len(results) > ef:
                    heapq.heappop(results)

    result_list = [(-neg_s, idx) for neg_s, idx in results if idx != exclude]
    result_list.sort(reverse=True)
    return [idx for _, idx in result_list], n_dist


def multi_layer_search(query_idx, layer_graphs, embeddings_normed,
                       entry_point, ef=30, top_k=10):
    """Search a multi-layer HNSW-style graph.

    Starts at the top layer, greedily descends. Each layer uses ef=1
    (pure greedy) except the bottom layer which uses the full ef beam.
    """
    query_vec = embeddings_normed[query_idx]
    total_dist = 0
    current_entry = entry_point

    # Top layers: greedy descent (ef=1)
    for layer_graph in reversed(layer_graphs[1:]):
        results, n_dist = greedy_search(
            query_vec, current_entry, layer_graph, embeddings_normed,
            ef=1, exclude=query_idx)
        total_dist += n_dist
        if results:
            current_entry = results[0]

    # Layer 0: full beam search
    results, n_dist = greedy_search(
        query_vec, current_entry, layer_graphs[0], embeddings_normed,
        ef=ef, exclude=query_idx)
    total_dist += n_dist

    return results[:top_k], total_dist


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
        description="Compare HNSW promotion strategies: random vs multi-address")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="Number of query points to evaluate")
    parser.add_argument("--ef", type=int, default=100,
                        help="Beam width for layer 0 search")
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
    bucket_ids = np.asarray(clf.get_bucket_ids())
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles_all))

    embeddings = embeddings_all[dedup_mask]
    n = len(embeddings)
    print(f"  {len(titles_all)} -> {n} after dedup")

    # Normalize once for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # ── Bridge scores ────────────────────────────────────────────────────
    print("\nComputing bridge scores...")

    # Simple LSH
    clf2 = RustClassifier(
        embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf2.fit(embeddings)
    centroid_sims = np.array(clf2.get_centroid_similarities())
    simple_rank = np.argsort(centroid_sims)  # ascending = strongest bridges

    # Multi-address
    print(f"Building PCA tree (depth={args.max_depth})...")
    t0 = time.time()
    tree = build_pca_tree(embeddings, args.max_depth)
    print(f"  Built in {time.time() - t0:.1f}s")

    ma_result = extract_boundary_persistence(tree, margin_pct=args.margin_pct)
    boundary_depths = ma_result['boundary_depths']

    ma_score = np.zeros(n, dtype=np.float64)
    for pt_idx, depths in boundary_depths.items():
        ma_score[pt_idx] = sum(args.max_depth - d for d in depths)

    ma_rank = np.lexsort((centroid_sims, -ma_score))

    # ── Ground truth: brute-force kNN ────────────────────────────────────
    print("\nComputing brute-force k=10 nearest neighbors (ground truth)...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=11, metric='cosine')
    nn.fit(embeddings)
    _, true_knn = nn.kneighbors(embeddings)
    true_knn = true_knn[:, 1:]  # drop self
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── HNSW parameters ─────────────────────────────────────────────────
    M_conn = 16
    mL = 1.0 / np.log(M_conn)

    # ── Build layer 0 variants ───────────────────────────────────────────
    # Bridge points for selective symmetrization: top ~6% by multi-address
    # (matching the ~n/M that would be promoted to layer 1)
    n_bridges_for_sym = int(n / M_conn)
    bridge_indices_for_sym = ma_rank[:n_bridges_for_sym]

    print(f"\nBuilding layer 0 graphs (k=32, {n_bridges_for_sym} bridge points)...")
    t0 = time.time()
    layer0_variants, edge_counts = build_layer0_graphs(
        embeddings, k=32, bridge_indices=bridge_indices_for_sym)
    print(f"  Done in {time.time() - t0:.1f}s")
    for gname, count in edge_counts.items():
        print(f"  {gname}: {count} edges ({count/n:.1f} avg degree)")

    # ── Select query points ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n_queries = min(args.n_queries, n)
    query_indices = rng.choice(n, n_queries, replace=False)
    print(f"\nEvaluating {n_queries} queries, ef={args.ef}")

    # ── Layer assignments ────────────────────────────────────────────────
    # Random (HNSW default): layer = floor(-ln(U) * mL)
    random_layers = np.floor(-np.log(rng.uniform(size=n) + 1e-15) * mL).astype(int)

    # Ranked: top points get promoted, matching exponential distribution
    def rank_to_layers(ranked_indices, n_total):
        layers = np.zeros(n_total, dtype=int)
        lyr = 0
        while True:
            count_at_layer = int(n_total / (M_conn ** (lyr + 1)))
            if count_at_layer < 1:
                break
            layers[ranked_indices[:count_at_layer]] = lyr + 1
            lyr += 1
        return layers

    ma_layers = rank_to_layers(ma_rank, n)

    max_layer = max(int(random_layers.max()), int(ma_layers.max()))
    print(f"\nLayer distribution (M={M_conn}):")
    for lyr in range(max_layer + 1):
        r_count = int(np.sum(random_layers >= lyr))
        m_count = int(np.sum(ma_layers >= lyr))
        print(f"  Layer {lyr}: random={r_count}, multi-addr={m_count}")

    # ── Evaluate combinations ────────────────────────────────────────────
    # Test: promotion strategy x graph type
    experiments = [
        ("Random + bidir",         random_layers, "bidirectional"),
        ("Random + pruned-bidir",  random_layers, "pruned-bidir"),
        ("MA + directed",          ma_layers,     "directed"),
        ("MA + bridge-bidir",      ma_layers,     "bridge-bidir"),
        ("MA + pruned-bridge",     ma_layers,     "pruned-bridge"),
        ("MA + pruned-bidir",      ma_layers,     "pruned-bidir"),
        ("MA + bidir",             ma_layers,     "bidirectional"),
    ]

    rows = []
    for exp_name, point_layers, graph_key in experiments:
        layer0 = layer0_variants[graph_key]
        n_layers = int(point_layers.max()) + 1
        n_edges = edge_counts[graph_key]

        print(f"\n--- {exp_name} ({n_layers} layers, {n_edges} L0 edges) ---")

        # Build upper-layer graphs (always bidirectional for upper layers)
        layer_graphs = [layer0]
        for lyr in range(1, n_layers):
            promoted = np.where(point_layers >= lyr)[0]
            if len(promoted) < 2:
                layer_graphs.append({})
                continue
            graph = build_upper_layer_graph(
                emb_normed, promoted,
                k=min(M_conn, len(promoted) - 1))
            layer_graphs.append(graph)

        # Entry point: highest-layer point nearest to dataset centroid
        top_layer = n_layers - 1
        top_points = np.where(point_layers >= top_layer)[0]
        centroid = emb_normed.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-10)
        entry_sims = emb_normed[top_points] @ centroid
        entry_point = int(top_points[np.argmax(entry_sims)])

        # Search
        recalls = []
        dist_counts = []

        for q in query_indices:
            true_set = set(true_knn[q].tolist())
            results, n_dist = multi_layer_search(
                q, layer_graphs, emb_normed,
                entry_point, ef=args.ef, top_k=10)
            found = len(set(results) & true_set)
            recalls.append(found / 10.0)
            dist_counts.append(n_dist)

        mean_recall = np.mean(recalls)
        mean_dist = np.mean(dist_counts)
        efficiency = mean_recall / (mean_dist / 1000) if mean_dist > 0 else 0

        print(f"  recall@10={mean_recall:.3f}  avg_dist={mean_dist:.0f}  "
              f"eff={efficiency:.2f}")

        rows.append([
            exp_name,
            f"{n_edges}",
            f"{mean_recall:.3f}",
            f"{mean_dist:.0f}",
            f"{efficiency:.2f}",
        ])

    print_table(
        "HNSW Promotion x Graph Symmetry",
        ["Strategy", "L0 Edges", "Recall@10", "Avg Dist", "Recall/1k Dist"],
        rows,
    )

    # ── Real hnswlib comparison ──────────────────────────────────────────
    # Stock hnswlib with different insertion orders.
    # HNSW graph quality depends on insertion order because early points
    # form the skeleton. Inserting bridges first = better skeleton.
    import hnswlib

    M_hnsw = 16
    ef_construction = 200
    dim = embeddings.shape[1]

    print(f"\n{'=' * 78}")
    print(f"  Real hnswlib comparison (M={M_hnsw}, ef_construction={ef_construction})")
    print(f"{'=' * 78}")

    def build_and_query_hnsw(embs, insertion_order, ef_values, query_idx,
                              true_knn_arr, label):
        """Build hnswlib index with given insertion order, query at multiple ef."""
        idx = hnswlib.Index(space='cosine', dim=dim)
        idx.init_index(max_elements=len(embs), M=M_hnsw,
                       ef_construction=ef_construction)
        idx.set_num_threads(1)  # single-thread for fair timing

        # Insert in specified order
        ordered_embs = embs[insertion_order]
        ordered_ids = insertion_order
        idx.add_items(ordered_embs, ordered_ids)

        results = []
        for ef in ef_values:
            idx.set_ef(ef)

            t0 = time.time()
            labels, distances = idx.knn_query(embs[query_idx], k=11,
                                               num_threads=1)
            elapsed = time.time() - t0

            # Compute recall
            recalls = []
            for i, q in enumerate(query_idx):
                true_set = set(true_knn_arr[q].tolist())
                found_set = set(labels[i].tolist()) - {q}
                recalls.append(len(true_set & found_set) / 10.0)

            mean_recall = np.mean(recalls)
            qps = len(query_idx) / elapsed if elapsed > 0 else 0
            results.append((ef, mean_recall, qps))
            print(f"  {label:30s} ef={ef:>4d}: "
                  f"recall@10={mean_recall:.3f}  "
                  f"QPS={qps:.0f}")

        return results

    # Insertion orders to test
    random_order = rng.permutation(n).astype(np.int64)
    # Bridges first: top multi-address points first, then rest
    bridges_first = np.concatenate([
        ma_rank[:n_bridges_for_sym],
        np.setdiff1d(np.arange(n), ma_rank[:n_bridges_for_sym])
    ]).astype(np.int64)
    # Bridges first with shuffled non-bridges
    rest = np.setdiff1d(np.arange(n), ma_rank[:n_bridges_for_sym])
    rng.shuffle(rest)
    bridges_first_shuffled = np.concatenate([
        ma_rank[:n_bridges_for_sym], rest
    ]).astype(np.int64)

    ef_values = [10, 20, 50, 100, 200, 400]

    print(f"\n  Testing {len(query_indices)} queries across ef={ef_values}")
    print()

    all_results = {}
    for label, order in [("Stock HNSW (random order)", random_order),
                          ("Bridges first", bridges_first_shuffled)]:
        all_results[label] = build_and_query_hnsw(
            embeddings, order, ef_values, query_indices, true_knn, label)
        print()

    # Summary table
    hnsw_rows = []
    for ef in ef_values:
        row = [str(ef)]
        for label in all_results:
            for ef_r, recall, qps in all_results[label]:
                if ef_r == ef:
                    row.append(f"{recall:.3f} ({qps:.0f} QPS)")
        hnsw_rows.append(row)

    print_table(
        "hnswlib Recall@10 by ef (insertion order comparison)",
        ["ef", "Stock HNSW", "Bridges First"],
        hnsw_rows,
    )

    print()


if __name__ == "__main__":
    main()
