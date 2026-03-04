"""
Clustering experiments using dyf features.

Compares different approaches to clustering embeddings using density-based features
from DensityClassifier, tracking Gini coefficient and average similarity.

Goal: Find clustering that avoids "big blob in center" problem.
"""

import numpy as np
import polars as pl
from collections import defaultdict, Counter
from pathlib import Path


def gini_coefficient(sizes: list) -> float:
    """Compute Gini coefficient of cluster sizes. 0 = perfect equality, 1 = max inequality."""
    sizes = np.array(sorted(sizes))
    n = len(sizes)
    if n == 0 or sizes.sum() == 0:
        return 0.0
    cumsum = np.cumsum(sizes)
    return (2 * np.sum((np.arange(1, n+1) * sizes)) - (n + 1) * sizes.sum()) / (n * sizes.sum())


def avg_within_cluster_similarity(embeddings: np.ndarray, cluster_assignments: np.ndarray) -> float:
    """Compute average within-cluster cosine similarity."""
    unique_clusters = np.unique(cluster_assignments)
    total_sim = 0.0
    total_pairs = 0

    for c in unique_clusters:
        mask = cluster_assignments == c
        cluster_embs = embeddings[mask]
        n = len(cluster_embs)
        if n < 2:
            continue

        # Sample if cluster is large
        if n > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, 500, replace=False)
            cluster_embs = cluster_embs[idx]
            n = 500

        # Compute pairwise similarities
        sim_matrix = cluster_embs @ cluster_embs.T
        # Upper triangle only (exclude diagonal)
        upper_tri = np.triu_indices(n, k=1)
        sims = sim_matrix[upper_tri]
        total_sim += sims.sum()
        total_pairs += len(sims)

    return total_sim / total_pairs if total_pairs > 0 else 0.0


def largest_cluster_fraction(cluster_assignments: np.ndarray) -> float:
    """Fraction of points in the largest cluster."""
    counts = Counter(cluster_assignments)
    max_count = max(counts.values())
    return max_count / len(cluster_assignments)


def evaluate_clustering(name: str, embeddings: np.ndarray, assignments: np.ndarray):
    """Evaluate a clustering and print metrics."""
    unique = np.unique(assignments)
    sizes = [np.sum(assignments == c) for c in unique]

    gini = gini_coefficient(sizes)
    avg_sim = avg_within_cluster_similarity(embeddings, assignments)
    largest_frac = largest_cluster_fraction(assignments)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Clusters: {len(unique)}")
    print(f"  Gini coefficient: {gini:.3f}")
    print(f"  Avg within-cluster similarity: {avg_sim:.3f}")
    print(f"  Largest cluster: {max(sizes):,} ({largest_frac:.1%})")
    print(f"  Smallest cluster: {min(sizes):,}")
    print(f"  Median cluster size: {int(np.median(sizes)):,}")

    # Size distribution
    print(f"  Size distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th percentile: {int(np.percentile(sizes, p)):,}")

    return {
        'name': name,
        'n_clusters': len(unique),
        'gini': gini,
        'avg_sim': avg_sim,
        'largest_frac': largest_frac,
        'sizes': sizes
    }


def experiment_direct_buckets(embeddings, classifier, n_clusters=15):
    """Use top N LSH buckets by size, rest go to 'other'."""
    bucket_ids = classifier.get_bucket_ids()

    # Count bucket sizes
    bucket_counts = Counter(bucket_ids)
    top_buckets = [b for b, _ in bucket_counts.most_common(n_clusters - 1)]
    top_set = set(top_buckets)

    # Assign cluster IDs
    assignments = np.zeros(len(bucket_ids), dtype=np.int32)
    bucket_to_cluster = {b: i for i, b in enumerate(top_buckets)}
    other_cluster = n_clusters - 1

    for i, bid in enumerate(bucket_ids):
        if bid in top_set:
            assignments[i] = bucket_to_cluster[bid]
        else:
            assignments[i] = other_cluster

    return assignments


def experiment_bridge_communities(embeddings, classifier, bridge_analysis, min_community_size=100):
    """Build bucket graph from bridges, find connected components."""
    import networkx as nx

    bucket_ids = classifier.get_bucket_ids()
    unique_buckets = set(bucket_ids)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(unique_buckets)

    # Add edges from bridge connections
    top_pairs = bridge_analysis.top_connected_pairs(500)
    for b1, b2, count in top_pairs:
        if b1 in unique_buckets and b2 in unique_buckets:
            G.add_edge(b1, b2, weight=count)

    # Find communities using Louvain
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, weight='weight', seed=42)
    except ImportError:
        # Fallback to connected components
        communities = list(nx.connected_components(G))

    # Sort communities by size
    communities = sorted(communities, key=lambda x: -len(x))

    # Map buckets to community IDs
    bucket_to_community = {}
    for i, comm in enumerate(communities):
        for bucket in comm:
            bucket_to_community[bucket] = i

    # Assign points
    assignments = np.array([bucket_to_community.get(bid, len(communities)) for bid in bucket_ids], dtype=np.int32)

    return assignments


def experiment_bucket_agglomerative(embeddings, classifier, n_clusters=15):
    """Agglomerative clustering on bucket centroids."""
    from sklearn.cluster import AgglomerativeClustering

    bucket_ids = classifier.get_bucket_ids()

    # Compute bucket centroids
    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    bucket_list = list(bucket_to_indices.keys())
    centroids = np.array([embeddings[bucket_to_indices[b]].mean(axis=0) for b in bucket_list])

    # Cluster centroids
    agg = AgglomerativeClustering(n_clusters=min(n_clusters, len(bucket_list)), metric='cosine', linkage='average')
    bucket_clusters = agg.fit_predict(centroids)

    # Map buckets to clusters
    bucket_to_cluster = {b: bucket_clusters[i] for i, b in enumerate(bucket_list)}

    # Assign points
    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)

    return assignments


def experiment_isolation_stratified(embeddings, classifier, n_strata=10):
    """Stratify by isolation score, then subdivide by bucket."""
    bucket_ids = np.array(classifier.get_bucket_ids())
    isolation_scores = np.array(classifier.get_isolation_scores())

    # Create isolation strata
    percentiles = np.percentile(isolation_scores, np.linspace(0, 100, n_strata + 1))
    strata = np.digitize(isolation_scores, percentiles[1:-1])

    # Within each stratum, use top buckets
    assignments = np.zeros(len(bucket_ids), dtype=np.int32)
    cluster_id = 0

    for s in range(n_strata):
        mask = strata == s
        stratum_buckets = bucket_ids[mask]
        bucket_counts = Counter(stratum_buckets)

        # Top 2 buckets per stratum get own cluster
        top_buckets = [b for b, _ in bucket_counts.most_common(2)]

        for b in top_buckets:
            bucket_mask = (bucket_ids == b) & mask
            assignments[bucket_mask] = cluster_id
            cluster_id += 1

        # Rest of stratum
        other_mask = mask & ~np.isin(bucket_ids, top_buckets)
        if other_mask.sum() > 0:
            assignments[other_mask] = cluster_id
            cluster_id += 1

    return assignments


def experiment_stability_weighted(embeddings, classifier, n_clusters=15):
    """Weight buckets by stability score, cluster high-stability separately."""
    bucket_ids = np.array(classifier.get_bucket_ids())
    stability_scores = np.array(classifier.get_stability_scores())

    # High stability points (top 50%) - cluster by bucket
    high_stability_mask = stability_scores >= np.median(stability_scores)

    bucket_counts_high = Counter(bucket_ids[high_stability_mask])
    top_buckets = [b for b, _ in bucket_counts_high.most_common(n_clusters - 2)]

    assignments = np.full(len(bucket_ids), n_clusters - 1, dtype=np.int32)  # default to "other"

    bucket_to_cluster = {b: i for i, b in enumerate(top_buckets)}

    for i, (bid, stab) in enumerate(zip(bucket_ids, stability_scores)):
        if bid in bucket_to_cluster and stab >= np.median(stability_scores):
            assignments[i] = bucket_to_cluster[bid]
        elif stab < np.percentile(stability_scores, 25):
            assignments[i] = n_clusters - 2  # low stability cluster

    return assignments


def experiment_density_hierarchical(embeddings, classifier, bridge_analysis, target_clusters=15):
    """Hierarchically merge buckets based on bridge density."""
    bucket_ids = classifier.get_bucket_ids()
    bucket_sizes = classifier.get_bucket_sizes()

    # Start with each bucket as its own cluster
    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Get bridge connections as merge candidates
    top_pairs = bridge_analysis.top_connected_pairs(1000)

    # Union-find for merging
    parent = {b: b for b in bucket_to_indices.keys()}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge buckets with strong bridge connections
    n_buckets = len(bucket_to_indices)
    for b1, b2, count in top_pairs:
        if b1 in parent and b2 in parent:
            # Only merge if it reduces cluster count toward target
            current_clusters = len(set(find(b) for b in parent.keys()))
            if current_clusters > target_clusters:
                union(b1, b2)

    # Map to final cluster IDs
    root_to_cluster = {}
    cluster_id = 0
    for b in parent.keys():
        root = find(b)
        if root not in root_to_cluster:
            root_to_cluster[root] = cluster_id
            cluster_id += 1

    # Assign points
    assignments = np.array([root_to_cluster[find(bid)] for bid in bucket_ids], dtype=np.int32)

    return assignments


def experiment_centroid_similarity_split(embeddings, classifier, n_clusters=15):
    """Split buckets by centroid similarity - low similarity items form separate clusters."""
    bucket_ids = np.array(classifier.get_bucket_ids())
    centroid_sims = np.array(classifier.get_centroid_similarities())

    # Points with low centroid similarity are "edge" points
    sim_threshold = np.percentile(centroid_sims, 30)  # bottom 30%

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Sort buckets by size
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    assignments = np.zeros(len(bucket_ids), dtype=np.int32)
    cluster_id = 0

    # Top buckets: split into core (high sim) and edge (low sim)
    for bid, indices in sorted_buckets[:n_clusters // 2]:
        indices = np.array(indices)
        sims = centroid_sims[indices]

        core_mask = sims >= sim_threshold
        edge_mask = ~core_mask

        if core_mask.sum() > 0:
            assignments[indices[core_mask]] = cluster_id
            cluster_id += 1

        if edge_mask.sum() > 0:
            assignments[indices[edge_mask]] = cluster_id
            cluster_id += 1

    # Rest go to remaining clusters based on similarity bands
    remaining_mask = assignments == 0
    remaining_sims = centroid_sims[remaining_mask]

    if remaining_mask.sum() > 0:
        # Split remaining by similarity quartiles
        remaining_indices = np.where(remaining_mask)[0]
        quartiles = np.percentile(remaining_sims, [25, 50, 75])

        for i, idx in enumerate(remaining_indices):
            sim = centroid_sims[idx]
            if sim < quartiles[0]:
                assignments[idx] = cluster_id
            elif sim < quartiles[1]:
                assignments[idx] = cluster_id + 1
            elif sim < quartiles[2]:
                assignments[idx] = cluster_id + 2
            else:
                assignments[idx] = cluster_id + 3

    return assignments


def experiment_bridge_merge_controlled(embeddings, classifier, bridge_analysis, target_clusters=15):
    """Bridge communities but force merge small ones until we hit target."""
    import networkx as nx

    bucket_ids = np.array(classifier.get_bucket_ids())
    unique_buckets = set(bucket_ids)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(unique_buckets)

    top_pairs = bridge_analysis.top_connected_pairs(500)
    for b1, b2, count in top_pairs:
        if b1 in unique_buckets and b2 in unique_buckets:
            G.add_edge(b1, b2, weight=count)

    # Start with connected components
    components = list(nx.connected_components(G))

    # Compute component sizes
    bucket_to_component = {}
    for i, comp in enumerate(components):
        for b in comp:
            bucket_to_component[b] = i

    # Get component sizes in terms of points
    comp_sizes = defaultdict(int)
    for bid in bucket_ids:
        comp_id = bucket_to_component.get(bid, -1)
        comp_sizes[comp_id] += 1

    # Sort components by size
    sorted_comps = sorted(comp_sizes.items(), key=lambda x: -x[1])

    # Take top target_clusters-1, merge rest
    top_comps = set(c for c, _ in sorted_comps[:target_clusters - 1])

    assignments = np.zeros(len(bucket_ids), dtype=np.int32)
    comp_to_cluster = {c: i for i, (c, _) in enumerate(sorted_comps[:target_clusters - 1])}

    for i, bid in enumerate(bucket_ids):
        comp_id = bucket_to_component.get(bid, -1)
        if comp_id in top_comps:
            assignments[i] = comp_to_cluster[comp_id]
        else:
            assignments[i] = target_clusters - 1  # "other"

    return assignments


def experiment_bucket_size_balanced(embeddings, classifier, target_clusters=15):
    """Greedily assign buckets to clusters to balance sizes."""
    bucket_ids = np.array(classifier.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Sort buckets by size descending
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    # Greedy assignment: always add to smallest cluster
    cluster_sizes = [0] * target_clusters
    bucket_to_cluster = {}

    for bid, indices in sorted_buckets:
        # Find smallest cluster
        min_cluster = np.argmin(cluster_sizes)
        bucket_to_cluster[bid] = min_cluster
        cluster_sizes[min_cluster] += len(indices)

    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)

    return assignments


def experiment_similarity_then_merge(embeddings, classifier, sim_threshold=0.4, max_clusters=12):
    """Create natural clusters by similarity, then merge smallest until we hit max_clusters."""
    bucket_ids = np.array(classifier.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Compute bucket centroids
    centroids = {b: embeddings[indices].mean(axis=0) for b, indices in bucket_to_indices.items()}

    # Sort buckets by size descending
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    # Phase 1: Natural clustering by similarity threshold
    clusters = []  # list of [centroid, size, bucket_list]
    bucket_to_cluster = {}

    for bid, indices in sorted_buckets:
        bucket_centroid = centroids[bid]
        bucket_size = len(indices)

        if not clusters:
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = 0
            continue

        sims = [np.dot(bucket_centroid, c[0]) for c in clusters]
        best_cluster = np.argmax(sims)
        best_sim = sims[best_cluster]

        if best_sim >= sim_threshold:
            bucket_to_cluster[bid] = best_cluster
            old_size = clusters[best_cluster][1]
            new_size = old_size + bucket_size
            clusters[best_cluster][0] = (
                clusters[best_cluster][0] * old_size + bucket_centroid * bucket_size
            ) / new_size
            clusters[best_cluster][1] = new_size
            clusters[best_cluster][2].append(bid)
        else:
            new_cluster_id = len(clusters)
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = new_cluster_id

    print(f"  Phase 1: {len(clusters)} natural clusters (threshold={sim_threshold})")

    # Phase 2: Merge smallest clusters until we hit max_clusters
    while len(clusters) > max_clusters:
        # Find smallest cluster
        sizes = [c[1] for c in clusters]
        smallest_idx = np.argmin(sizes)
        smallest = clusters[smallest_idx]

        # Find most similar other cluster
        sims = [np.dot(smallest[0], c[0]) if i != smallest_idx else -1
                for i, c in enumerate(clusters)]
        merge_target = np.argmax(sims)

        # Merge smallest into target
        target = clusters[merge_target]
        old_size = target[1]
        new_size = old_size + smallest[1]
        target[0] = (target[0] * old_size + smallest[0] * smallest[1]) / new_size
        target[1] = new_size
        target[2].extend(smallest[2])

        # Update bucket assignments
        for bid in smallest[2]:
            bucket_to_cluster[bid] = merge_target

        # Remove smallest cluster
        clusters.pop(smallest_idx)

        # Reindex clusters after removal
        for i, c in enumerate(clusters):
            for bid in c[2]:
                bucket_to_cluster[bid] = i

    print(f"  Phase 2: Merged to {len(clusters)} clusters")

    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)
    return assignments


def experiment_similarity_threshold_only(embeddings, classifier, sim_threshold=0.4):
    """Cluster buckets purely by similarity - no target cluster count."""
    bucket_ids = np.array(classifier.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Compute bucket centroids
    centroids = {b: embeddings[indices].mean(axis=0) for b, indices in bucket_to_indices.items()}

    # Sort buckets by size descending (largest become seeds)
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    # Greedy assignment: each bucket joins most similar existing cluster OR starts new one
    clusters = []  # list of (centroid, size, bucket_list)
    bucket_to_cluster = {}

    for bid, indices in sorted_buckets:
        bucket_centroid = centroids[bid]
        bucket_size = len(indices)

        if not clusters:
            # First bucket starts first cluster
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = 0
            continue

        # Find most similar cluster
        sims = [np.dot(bucket_centroid, c[0]) for c in clusters]
        best_cluster = np.argmax(sims)
        best_sim = sims[best_cluster]

        if best_sim >= sim_threshold:
            # Join existing cluster
            bucket_to_cluster[bid] = best_cluster
            # Update centroid (weighted average)
            old_size = clusters[best_cluster][1]
            new_size = old_size + bucket_size
            clusters[best_cluster][0] = (
                clusters[best_cluster][0] * old_size + bucket_centroid * bucket_size
            ) / new_size
            clusters[best_cluster][1] = new_size
            clusters[best_cluster][2].append(bid)
        else:
            # Start new cluster
            new_cluster_id = len(clusters)
            clusters.append([bucket_centroid.copy(), bucket_size, [bid]])
            bucket_to_cluster[bid] = new_cluster_id

    print(f"  Similarity threshold {sim_threshold} -> {len(clusters)} clusters")

    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)
    return assignments


def experiment_similarity_constrained_balance(embeddings, classifier, target_clusters=15, min_sim_threshold=0.35):
    """Balance sizes but only merge buckets if their centroids are similar."""
    bucket_ids = np.array(classifier.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Compute bucket centroids
    bucket_list = list(bucket_to_indices.keys())
    centroids = {b: embeddings[bucket_to_indices[b]].mean(axis=0) for b in bucket_list}

    # Sort buckets by size descending
    sorted_buckets = sorted(bucket_to_indices.items(), key=lambda x: -len(x[1]))

    # Initialize clusters with largest buckets
    cluster_centroids = []
    cluster_sizes = []
    bucket_to_cluster = {}

    for bid, indices in sorted_buckets[:target_clusters]:
        cluster_id = len(cluster_centroids)
        bucket_to_cluster[bid] = cluster_id
        cluster_centroids.append(centroids[bid].copy())
        cluster_sizes.append(len(indices))

    # Assign remaining buckets to most similar cluster that isn't too big
    target_size = len(bucket_ids) / target_clusters

    for bid, indices in sorted_buckets[target_clusters:]:
        bucket_centroid = centroids[bid]

        # Compute similarity to each cluster
        sims = [np.dot(bucket_centroid, cc) for cc in cluster_centroids]

        # Sort clusters by similarity (descending)
        sorted_clusters = sorted(range(len(sims)), key=lambda x: -sims[x])

        # Pick first cluster that meets similarity threshold and isn't too large
        assigned = False
        for cluster_id in sorted_clusters:
            if sims[cluster_id] >= min_sim_threshold:
                # Check if adding would make it too big (2x target)
                if cluster_sizes[cluster_id] + len(indices) <= target_size * 2.5:
                    bucket_to_cluster[bid] = cluster_id
                    cluster_sizes[cluster_id] += len(indices)
                    # Update centroid
                    n = cluster_sizes[cluster_id]
                    cluster_centroids[cluster_id] = (
                        cluster_centroids[cluster_id] * (n - len(indices)) + bucket_centroid * len(indices)
                    ) / n
                    assigned = True
                    break

        if not assigned:
            # Assign to smallest cluster
            min_cluster = np.argmin(cluster_sizes)
            bucket_to_cluster[bid] = min_cluster
            cluster_sizes[min_cluster] += len(indices)

    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)
    return assignments


def experiment_isolation_buckets_hybrid(embeddings, classifier, target_clusters=15):
    """Combine isolation stratification with bucket coherence."""
    bucket_ids = np.array(classifier.get_bucket_ids())
    isolation_scores = np.array(classifier.get_isolation_scores())

    bucket_to_indices = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        bucket_to_indices[bid].append(i)

    # Compute mean isolation per bucket
    bucket_isolations = {
        bid: np.mean(isolation_scores[indices])
        for bid, indices in bucket_to_indices.items()
    }

    # Sort buckets by mean isolation
    sorted_by_isolation = sorted(bucket_isolations.items(), key=lambda x: x[1])

    # Divide into isolation bands and assign buckets to clusters
    n_buckets = len(sorted_by_isolation)
    buckets_per_cluster = max(1, n_buckets // target_clusters)

    bucket_to_cluster = {}
    cluster_id = 0

    for i, (bid, iso) in enumerate(sorted_by_isolation):
        if i > 0 and i % buckets_per_cluster == 0 and cluster_id < target_clusters - 1:
            cluster_id += 1
        bucket_to_cluster[bid] = cluster_id

    assignments = np.array([bucket_to_cluster[bid] for bid in bucket_ids], dtype=np.int32)
    return assignments


def main():
    from dyf import DensityClassifier

    parquet_path = Path("/Users/jdonaldson/Projects/dyf/demo/gudid_50k.parquet")

    print(f"Loading embeddings from {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    embeddings = np.array(df['embedding'].to_list(), dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    print(f"Loaded {len(embeddings):,} embeddings")

    # Run classifier
    print("\nRunning DensityClassifier...")
    classifier = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    classifier.fit(embeddings)
    print(f"  {classifier.report()}")

    # Get bridge analysis
    print("Analyzing bridges...")
    bridge_analysis = classifier.analyze_bridges(embeddings)
    print(f"  {len(bridge_analysis.bridge_indices)} bridge points")

    results = []

    # Experiment 1: Direct top buckets
    print("\n" + "="*60)
    print("EXPERIMENT 1: Direct top buckets (top 14 + other)")
    assignments = experiment_direct_buckets(embeddings, classifier, n_clusters=15)
    results.append(evaluate_clustering("Direct Top Buckets", embeddings, assignments))

    # Experiment 2: Bridge community detection
    print("\n" + "="*60)
    print("EXPERIMENT 2: Bridge-based community detection")
    assignments = experiment_bridge_communities(embeddings, classifier, bridge_analysis)
    results.append(evaluate_clustering("Bridge Communities", embeddings, assignments))

    # Experiment 3: Agglomerative on bucket centroids
    print("\n" + "="*60)
    print("EXPERIMENT 3: Agglomerative clustering on bucket centroids")
    assignments = experiment_bucket_agglomerative(embeddings, classifier, n_clusters=15)
    results.append(evaluate_clustering("Bucket Agglomerative", embeddings, assignments))

    # Experiment 4: Isolation-stratified
    print("\n" + "="*60)
    print("EXPERIMENT 4: Isolation score stratified")
    assignments = experiment_isolation_stratified(embeddings, classifier, n_strata=5)
    results.append(evaluate_clustering("Isolation Stratified", embeddings, assignments))

    # Experiment 5: Density hierarchical merge
    print("\n" + "="*60)
    print("EXPERIMENT 5: Density-based hierarchical merge")
    assignments = experiment_density_hierarchical(embeddings, classifier, bridge_analysis, target_clusters=15)
    results.append(evaluate_clustering("Density Hierarchical", embeddings, assignments))

    # Experiment 6: Centroid similarity split
    print("\n" + "="*60)
    print("EXPERIMENT 6: Centroid similarity split")
    assignments = experiment_centroid_similarity_split(embeddings, classifier, n_clusters=15)
    results.append(evaluate_clustering("Centroid Sim Split", embeddings, assignments))

    # Experiment 7: Bridge merge controlled
    print("\n" + "="*60)
    print("EXPERIMENT 7: Bridge communities (controlled merge)")
    assignments = experiment_bridge_merge_controlled(embeddings, classifier, bridge_analysis, target_clusters=15)
    results.append(evaluate_clustering("Bridge Controlled", embeddings, assignments))

    # Experiment 8: Bucket size balanced
    print("\n" + "="*60)
    print("EXPERIMENT 8: Greedy bucket size balancing")
    assignments = experiment_bucket_size_balanced(embeddings, classifier, target_clusters=15)
    results.append(evaluate_clustering("Size Balanced", embeddings, assignments))

    # Experiment 9: Similarity-constrained balance
    print("\n" + "="*60)
    print("EXPERIMENT 9: Similarity-constrained balance")
    assignments = experiment_similarity_constrained_balance(embeddings, classifier, target_clusters=15)
    results.append(evaluate_clustering("Sim-Constrained", embeddings, assignments))

    # Experiment 10: Isolation-bucket hybrid
    print("\n" + "="*60)
    print("EXPERIMENT 10: Isolation-bucket hybrid")
    assignments = experiment_isolation_buckets_hybrid(embeddings, classifier, target_clusters=15)
    results.append(evaluate_clustering("Iso-Bucket Hybrid", embeddings, assignments))

    # Experiment 11-13: Similarity threshold only (no target K)
    for thresh in [0.3, 0.4, 0.5]:
        print("\n" + "="*60)
        print(f"EXPERIMENT: Similarity threshold = {thresh}")
        assignments = experiment_similarity_threshold_only(embeddings, classifier, sim_threshold=thresh)
        results.append(evaluate_clustering(f"SimThresh={thresh}", embeddings, assignments))

    # K=12 experiments
    print("\n" + "="*60)
    print("=" * 60)
    print("K=12 CLUSTER EXPERIMENTS (for 12-color visualization)")
    print("=" * 60)

    print("\n" + "="*60)
    print("EXPERIMENT: Sim-Constrained K=12")
    assignments = experiment_similarity_constrained_balance(embeddings, classifier, target_clusters=12)
    results.append(evaluate_clustering("SimConstr K=12", embeddings, assignments))

    print("\n" + "="*60)
    print("EXPERIMENT: Size Balanced K=12")
    assignments = experiment_bucket_size_balanced(embeddings, classifier, target_clusters=12)
    results.append(evaluate_clustering("SizeBal K=12", embeddings, assignments))

    print("\n" + "="*60)
    print("EXPERIMENT: Isolation Stratified K=12")
    assignments = experiment_isolation_stratified(embeddings, classifier, n_strata=4)  # 4 strata * 3 clusters each ≈ 12
    results.append(evaluate_clustering("IsoStrat K=12", embeddings, assignments))

    print("\n" + "="*60)
    print("EXPERIMENT: Natural clusters merged to K=12")
    assignments = experiment_similarity_then_merge(embeddings, classifier, sim_threshold=0.4, max_clusters=12)
    results.append(evaluate_clustering("NatMerge K=12", embeddings, assignments))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Clusters':>8} {'Gini':>8} {'AvgSim':>8} {'Largest%':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['n_clusters']:>8} {r['gini']:>8.3f} {r['avg_sim']:>8.3f} {r['largest_frac']:>9.1%}")

    # Best by metrics
    print("\n" + "-"*60)
    best_gini = min(results, key=lambda x: x['gini'])
    best_sim = max(results, key=lambda x: x['avg_sim'])
    best_balance = min(results, key=lambda x: x['largest_frac'])

    print(f"Best Gini (most equal sizes): {best_gini['name']} ({best_gini['gini']:.3f})")
    print(f"Best Avg Similarity: {best_sim['name']} ({best_sim['avg_sim']:.3f})")
    print(f"Best Balance (smallest largest): {best_balance['name']} ({best_balance['largest_frac']:.1%})")


if __name__ == "__main__":
    main()
