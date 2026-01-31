"""
Preprocess wiki data for ROG (Recursive Ontological Graph) Browser.
Generates UMAP coords, clusters, LLM labels, and bundled edges.

Run: python demo/rog_preprocess.py demo/wiki_simple_50k.parquet --sample 50000
"""

import argparse
import numpy as np
import polars as pl
import pandas as pd
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import umap
from datashader.bundling import hammer_bundle
import subprocess

# Import dyf for bridge detection via ROG ontology
import dyf

CLUSTER_LEVELS = (12, 25, 50, 100)
BUNDLE_ITERATIONS = 4


def compute_density_knn(embeddings: np.ndarray, num_bits: int = 12, min_k: int = 15, max_k: int = 100):
    """Build variable-k neighbor graph using dyf bucket density.

    Points in sparse LSH buckets get more neighbors (need global context),
    points in dense buckets get fewer (local structure is clear).

    Returns (knn_indices, knn_dists) padded to max_k, suitable for UMAP precomputed_knn.
    """
    from sklearn.neighbors import NearestNeighbors
    from dyf_rs import DensityClassifier

    n = len(embeddings)
    print(f"  Building density-aware kNN (min_k={min_k}, max_k={max_k}, bits={num_bits})...")

    # Step 1: Get bucket densities from dyf
    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=num_bits, seed=42)
    clf.fit(embeddings.astype(np.float32))
    bucket_sizes = np.array(clf.get_bucket_sizes())

    # Step 2: Map bucket size → local k (inverse: sparse → high k, dense → low k)
    log_sizes = np.log1p(bucket_sizes)
    log_min, log_max = log_sizes.min(), log_sizes.max()
    if log_max > log_min:
        # Normalize to [0, 1] where 0=smallest bucket, 1=largest
        density_frac = (log_sizes - log_min) / (log_max - log_min)
    else:
        density_frac = np.zeros(n)

    # Invert: dense points get low k, sparse points get high k
    local_k = (min_k + (max_k - min_k) * (1 - density_frac)).astype(int)
    local_k = np.clip(local_k, min_k, max_k)

    print(f"  Local k distribution: min={local_k.min()}, median={int(np.median(local_k))}, "
          f"max={local_k.max()}, mean={local_k.mean():.0f}")
    print(f"  Bucket size distribution: min={bucket_sizes.min()}, median={int(np.median(bucket_sizes))}, "
          f"max={bucket_sizes.max()}")

    # Step 3: Build full kNN with max_k
    print(f"  Computing {max_k}-NN graph...")
    nn = NearestNeighbors(n_neighbors=max_k, metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    dists, indices = nn.kneighbors(embeddings)

    # Step 4: Mask out neighbors beyond each point's local k
    # Duplicate the last valid neighbor (UMAP needs valid indices, not self-refs at dist 0)
    for i in range(n):
        k_i = local_k[i]
        if k_i < max_k:
            indices[i, k_i:] = indices[i, k_i - 1]
            dists[i, k_i:] = dists[i, k_i - 1]

    print(f"  Density-aware kNN complete")
    return indices, dists


def load_data(path: str, sample: int | None = None,
              n_neighbors: int = 15, densmap: bool = False):
    """Load parquet and run UMAP. Returns coords, titles, and embeddings."""
    print(f"Loading {path}...")
    df = pl.read_parquet(path)

    if sample and sample < len(df):
        df = df.sample(sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list())

    mode = "densmap" if densmap else "standard"
    print(f"Running UMAP on {len(titles)} points ({mode}, n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_jobs=-1,
        verbose=True,
        densmap=densmap,
    )
    coords_2d = reducer.fit_transform(embeddings)

    # Replace NaN coords (disconnected UMAP vertices) with nearest valid neighbor
    nan_mask = np.isnan(coords_2d).any(axis=1)
    if nan_mask.any():
        n_nan = nan_mask.sum()
        print(f"  Replacing {n_nan} NaN coordinates with nearest valid neighbors")
        from sklearn.neighbors import NearestNeighbors
        valid = ~nan_mask
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[valid])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        valid_coords = coords_2d[valid]
        coords_2d[nan_mask] = valid_coords[idx.ravel()]

    # Normalize to (0,0) center with robust scaling.
    # Median-center (outlier-robust), then scale by the max MAD across axes
    # to preserve aspect ratio. Result: ~68% of points within radius 1.
    median = np.nanmedian(coords_2d, axis=0)
    mad = np.nanmedian(np.abs(coords_2d - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords_2d = (coords_2d - median) / scale
    print(f"Normalized coords: median-centered, MAD scale={scale:.3f}, "
          f"range x=[{coords_2d[:,0].min():.1f}, {coords_2d[:,0].max():.1f}] "
          f"y=[{coords_2d[:,1].min():.1f}, {coords_2d[:,1].max():.1f}]")

    return coords_2d, titles, embeddings


def compute_hierarchical_clusters(coords_2d: np.ndarray, titles: list[str]):
    """Cluster using k-means then cut dendrogram at multiple levels.

    This is the legacy function - kept for backwards compatibility.
    See compute_dendrogram_hierarchy for the new dynamic approach.
    """
    print("Computing hierarchical clusters...")

    initial_k = min(100, len(coords_2d) // 50)
    kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10)
    kmeans.fit(coords_2d)

    Z = linkage(kmeans.cluster_centers_, method='ward')

    result = {'labels': {}, 'names': {}, 'centroids': {}}

    for n_clusters in CLUSTER_LEVELS:
        centroid_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        labels = np.array([centroid_labels[l] for l in kmeans.labels_])
        result['labels'][n_clusters] = labels

        centroids = []
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centroids.append(coords_2d[mask].mean(axis=0))
            else:
                centroids.append(np.array([0.0, 0.0]))
        result['centroids'][n_clusters] = np.array(centroids)

    # LLM label all levels with contrastive labeling
    for n_clusters in CLUSTER_LEVELS:
        print(f"Generating contrastive labels for {n_clusters} clusters...")
        result['names'][n_clusters] = label_clusters_with_llm(
            titles, result['labels'][n_clusters], n_clusters,
            centroids=result['centroids'][n_clusters],
            contrastive=True,
        )

    return result


def sample_representative_points(
    point_indices: list[int],
    embeddings: np.ndarray,
    n_samples: int = 10,
    method: str = "mixed",
) -> list[int]:
    """Sample representative points from a cluster using embeddings.

    Args:
        point_indices: Indices of points in this cluster
        embeddings: Full embedding matrix
        n_samples: Number of points to sample
        method: "central" (closest to centroid), "random", or "mixed" (half each)

    Returns:
        List of sampled point indices
    """
    if len(point_indices) <= n_samples:
        return point_indices

    pts = np.array(point_indices)
    cluster_embeddings = embeddings[pts]

    if method == "random":
        chosen = np.random.choice(len(pts), n_samples, replace=False)
        return pts[chosen].tolist()

    # Compute centroid in embedding space
    centroid = cluster_embeddings.mean(axis=0)

    # Compute distances to centroid
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    if method == "central":
        # Take points closest to centroid
        closest = np.argsort(distances)[:n_samples]
        return pts[closest].tolist()

    # Mixed: half central, half random
    n_central = n_samples // 2
    n_random = n_samples - n_central

    closest = np.argsort(distances)[:n_central]
    central_pts = set(closest)

    # Random from remaining
    remaining = [i for i in range(len(pts)) if i not in central_pts]
    if remaining and n_random > 0:
        random_choice = np.random.choice(remaining, min(n_random, len(remaining)), replace=False)
        return pts[np.concatenate([closest, random_choice])].tolist()

    return pts[closest].tolist()


def compute_dendrogram_hierarchy(
    coords_2d: np.ndarray,
    titles: list[str],
    embeddings: np.ndarray | None = None,
    model: str = "gemma2:9b",
):
    """Build dendrogram with labels at every internal node for dynamic cutting.

    Args:
        coords_2d: 2D UMAP coordinates
        titles: List of titles
        embeddings: Original high-dim embeddings for representative sampling (optional)
        model: Ollama model for label generation

    Returns:
        dict with:
        - 'Z': linkage matrix
        - 'kmeans_labels': mapping from points to micro-clusters
        - 'kmeans_centroids': micro-cluster centroids in 2D
        - 'node_labels': dict mapping node_id -> label string
        - 'node_titles': dict mapping node_id -> list of title indices under this node
        - 'node_centroids': dict mapping node_id -> (x, y) centroid
        - 'max_dist': maximum distance in dendrogram (for normalization)
    """
    from scipy.cluster.hierarchy import to_tree
    from collections import defaultdict

    print("Computing dendrogram hierarchy...")

    # Step 1: K-means on high-dim embeddings (semantic clustering, not spatial)
    initial_k = min(100, len(coords_2d) // 50)
    print(f"  K-means with k={initial_k} on embeddings...")
    kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    # Step 2: Hierarchical clustering on micro-cluster centroids (in embedding space)
    print("  Building dendrogram...")
    Z = linkage(kmeans.cluster_centers_, method='ward')

    # Step 3: Build tree structure and collect info for each node
    tree = to_tree(Z, rd=True)
    root, nodes = tree

    # Map point indices to their kmeans cluster
    point_to_micro = kmeans.labels_

    # For each micro-cluster, get the point indices
    micro_to_points = defaultdict(list)
    for i, micro in enumerate(point_to_micro):
        micro_to_points[micro].append(i)

    # Build node info: for each node, get all point indices under it
    n_micro = initial_k
    node_points = {}  # node_id -> list of point indices
    node_centroids = {}  # node_id -> (x, y)

    def get_node_points(node):
        """Recursively get all point indices under a node."""
        if node.is_leaf():
            # Leaf node = micro-cluster
            micro_id = node.id
            return micro_to_points[micro_id]
        else:
            # Internal node = merge of left and right
            left_points = get_node_points(node.left)
            right_points = get_node_points(node.right)
            return left_points + right_points

    # Collect info for all nodes
    print("  Collecting node info...")
    for node in nodes:
        points = get_node_points(node)
        node_points[node.id] = points
        # Compute centroid
        if points:
            centroid = coords_2d[points].mean(axis=0)
            node_centroids[node.id] = (float(centroid[0]), float(centroid[1]))
        else:
            node_centroids[node.id] = (0.0, 0.0)

    # Step 4: Label internal nodes using RAG-style approach with representative sampling
    internal_nodes = [n for n in nodes if not n.is_leaf()]
    print(f"  Labeling {len(internal_nodes)} internal nodes with RAG sampling...")

    # Determine sampling method based on whether we have embeddings
    use_embeddings = embeddings is not None
    if use_embeddings:
        print("    Using embedding-based representative sampling")
    else:
        print("    Using random sampling (no embeddings provided)")

    node_labels = {}

    # Label leaf nodes (micro-clusters) with representative samples
    for node in nodes:
        if node.is_leaf():
            pts = node_points[node.id]
            if use_embeddings and len(pts) > 3:
                sample_pts = sample_representative_points(pts, embeddings, n_samples=5, method="central")
            else:
                sample_pts = pts[:5] if len(pts) > 5 else pts

            if sample_pts:
                sample_titles = [titles[p] for p in sample_pts[:3]]
                # Use short version of titles for leaf nodes
                node_labels[node.id] = sample_titles[0][:25] if sample_titles else f"Micro {node.id}"
            else:
                node_labels[node.id] = f"Micro {node.id}"

    # Label internal nodes with LLM using representative samples + TF-IDF keywords
    for i, node in enumerate(internal_nodes):
        if i % 10 == 0:
            print(f"    Labeling node {i+1}/{len(internal_nodes)}...")

        # Get sibling info (left vs right child for contrast)
        if node.left is None or node.right is None:
            continue  # Skip malformed nodes

        left_pts = node_points[node.left.id]
        right_pts = node_points[node.right.id]

        # Sample representative points from each side
        if use_embeddings:
            left_sample_pts = sample_representative_points(left_pts, embeddings, n_samples=12, method="mixed")
            right_sample_pts = sample_representative_points(right_pts, embeddings, n_samples=12, method="mixed")
        else:
            # Random sampling fallback
            if len(left_pts) > 12:
                left_sample_pts = np.random.choice(left_pts, 12, replace=False).tolist()
            else:
                left_sample_pts = left_pts
            if len(right_pts) > 12:
                right_sample_pts = np.random.choice(right_pts, 12, replace=False).tolist()
            else:
                right_sample_pts = right_pts

        left_sample = [titles[p] for p in left_sample_pts]
        right_sample = [titles[p] for p in right_sample_pts]

        # Compute TF-IDF keywords for left vs right child
        node_titles_list = [titles[p] for p in left_pts] + [titles[p] for p in right_pts]
        node_labels_arr = np.zeros(len(left_pts) + len(right_pts), dtype=int)
        node_labels_arr[len(left_pts):] = 1
        node_kw = compute_tfidf_keywords(node_titles_list, node_labels_arr, 2, top_k=5, min_df=1)
        left_keywords = [w for w, _ in node_kw.get(0, [])][:5]
        right_keywords = [w for w, _ in node_kw.get(1, [])][:5]

        # Generate label using LLM with contrastive TF-IDF prompt
        all_kw = left_keywords + right_keywords
        kw_str = f"\nDistinguishing keywords: {', '.join(all_kw)}" if all_kw else ""
        all_sample = left_sample + right_sample

        prompt = f"""Label this cluster of {len(left_pts) + len(right_pts)} items for a map visualization.
{kw_str}
Sample items:
{chr(10).join(f'- {t}' for t in all_sample)}

What broad TOPIC or THEME connects these items? Give a 2-4 word label like:
"European History", "Marine Biology", "Computer Science", "Popular Music", "Ancient Civilizations"

Be specific enough to distinguish from neighboring clusters. Avoid generic labels like "Historical Events" or "Various Topics".

Reply with ONLY the label, nothing else."""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            label = result.stdout.strip().split('\n')[0][:50]
            # Clean up common LLM artifacts
            label = label.strip('"\'').strip()
            node_labels[node.id] = label if label else f"Node {node.id}"
        except Exception as e:
            node_labels[node.id] = f"Node {node.id}"

    print("  Done labeling nodes.")

    # Get max distance for normalization
    max_dist = float(Z[:, 2].max()) if len(Z) > 0 else 1.0

    # Compute 2D centroids for each micro-cluster (for label placement)
    kmeans_centroids_2d = np.zeros((initial_k, 2))
    for micro_id in range(initial_k):
        pts = micro_to_points[micro_id]
        if pts:
            kmeans_centroids_2d[micro_id] = coords_2d[pts].mean(axis=0)

    return {
        'Z': Z,
        'kmeans_labels': point_to_micro,
        'kmeans_centroids': kmeans_centroids_2d,  # 2D centroids for label placement
        'node_labels': node_labels,
        'node_points': {k: v for k, v in node_points.items()},  # Convert to regular dict
        'node_centroids': node_centroids,
        'n_micro': n_micro,
        'max_dist': max_dist,
    }


def compute_tfidf_keywords(
    titles: list[str],
    labels: np.ndarray,
    n_clusters: int,
    top_k: int = 10,
    min_df: int = 2,
) -> dict[int, list[tuple[str, float]]]:
    """Compute TF-IDF keywords for each cluster using NLTK stop words.

    Args:
        titles: List of all titles
        labels: Cluster assignment for each title
        n_clusters: Number of clusters
        top_k: Number of top keywords to return per cluster
        min_df: Minimum document frequency (must appear in >= min_df clusters)

    Returns:
        Dict mapping cluster_id -> list of (word, tfidf_score) tuples
    """
    import re
    from collections import defaultdict
    import math

    # Get NLTK stop words
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

    # Add some additional stop words common in Wikipedia titles
    stop_words.update(['the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
                       'and', 'or', 'is', 'was', 'are', 'were', 'be', 'been',
                       'list', 'disambiguation', 'episode', 'season'])

    # Tokenize and build term frequencies per cluster
    def tokenize(text):
        # Simple tokenization: lowercase, split on non-alpha, filter short words
        words = re.findall(r'[a-z]+', text.lower())
        return [w for w in words if len(w) > 2 and w not in stop_words]

    # Group titles by cluster
    cluster_titles = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_titles[int(label)].append(titles[i])

    # Build vocabulary and document frequencies (df = number of clusters containing word)
    word_df = defaultdict(int)  # word -> number of clusters containing it
    cluster_word_counts = {}  # cluster -> {word: count}

    for cluster_id in range(n_clusters):
        word_counts = defaultdict(int)
        words_in_cluster = set()

        for title in cluster_titles[cluster_id]:
            for word in tokenize(title):
                word_counts[word] += 1
                words_in_cluster.add(word)

        cluster_word_counts[cluster_id] = word_counts

        for word in words_in_cluster:
            word_df[word] += 1

    # Filter vocabulary: must appear in >= min_df clusters (removes corpus-specific noise)
    # but also not in ALL clusters (removes ubiquitous words)
    vocab = {w for w, df in word_df.items() if min_df <= df < n_clusters}

    # Compute IDF with smoothing: log((N + 1) / (df + 1))
    N = n_clusters
    idf = {w: math.log((N + 1) / (word_df[w] + 1)) for w in vocab}

    # Compute TF-IDF for each cluster
    cluster_keywords = {}

    for cluster_id in range(n_clusters):
        word_counts = cluster_word_counts[cluster_id]
        total_words = sum(word_counts.values())

        if total_words == 0:
            cluster_keywords[cluster_id] = []
            continue

        # TF-IDF scores
        tfidf_scores = []
        for word in vocab:
            tf = word_counts.get(word, 0) / total_words
            score = tf * idf[word]
            if score > 0:
                tfidf_scores.append((word, score))

        # Sort by score and take top_k
        tfidf_scores.sort(key=lambda x: -x[1])
        cluster_keywords[cluster_id] = tfidf_scores[:top_k]

    return cluster_keywords


def disambiguate_cluster_names(
    names: list[str],
    titles: list[str],
    labels: np.ndarray,
) -> list[str]:
    """Append TF-IDF keywords to duplicate cluster names.

    After cut_dendrogram() assigns labels from dendrogram nodes, multiple clusters
    may end up with the same generic name (e.g., "Surgical Instruments").
    This function detects duplicates and appends distinguishing TF-IDF keywords.

    Args:
        names: List of cluster names (one per cluster)
        titles: List of all item titles
        labels: Cluster assignment for each title

    Returns:
        Updated names list with disambiguated duplicates
    """
    from collections import Counter

    name_counts = Counter(names)
    duplicates = {name for name, count in name_counts.items() if count > 1}

    if not duplicates:
        return names

    print(f"  Disambiguating {len(duplicates)} duplicate name(s)...")

    names = list(names)  # copy

    for dup_name in duplicates:
        # Find cluster IDs sharing this name
        dup_cluster_ids = [i for i, n in enumerate(names) if n == dup_name]
        print(f"    '{dup_name}' appears in clusters: {dup_cluster_ids}")

        # Build a sub-problem: only the points in these clusters, remapped to 0..k
        sub_titles = []
        sub_labels = []
        for new_id, cluster_id in enumerate(dup_cluster_ids):
            mask = labels == cluster_id
            cluster_titles = [titles[i] for i in range(len(titles)) if mask[i]]
            sub_titles.extend(cluster_titles)
            sub_labels.extend([new_id] * len(cluster_titles))

        sub_labels_arr = np.array(sub_labels, dtype=int)
        k = len(dup_cluster_ids)

        # Compute TF-IDF keywords across only the duplicate clusters
        sub_keywords = compute_tfidf_keywords(sub_titles, sub_labels_arr, k, top_k=5, min_df=1)

        # Assign top distinguishing keyword to each
        used_suffixes = set()
        for new_id, cluster_id in enumerate(dup_cluster_ids):
            kw_list = sub_keywords.get(new_id, [])
            suffix = None
            for word, score in kw_list:
                candidate = word
                if candidate not in used_suffixes:
                    suffix = candidate
                    used_suffixes.add(candidate)
                    break

            if suffix is None:
                # All single keywords taken; try two-word suffix
                for word, score in kw_list:
                    for word2, score2 in kw_list:
                        if word != word2:
                            candidate = f"{word}, {word2}"
                            if candidate not in used_suffixes:
                                suffix = candidate
                                used_suffixes.add(candidate)
                                break
                    if suffix:
                        break

            if suffix:
                names[cluster_id] = f"{dup_name} ({suffix})"
                print(f"      Cluster {cluster_id} -> '{names[cluster_id]}'")
            else:
                # Last resort: append cluster ID
                names[cluster_id] = f"{dup_name} ({cluster_id})"

    return names


def find_nearest_cluster(cluster_id: int, centroids: np.ndarray) -> int:
    """Find the nearest cluster to a given cluster (by centroid distance)."""
    target = centroids[cluster_id]
    min_dist = float('inf')
    nearest = 0
    for i, centroid in enumerate(centroids):
        if i != cluster_id:
            dist = np.linalg.norm(target - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest = i
    return nearest


def label_flat_clusters(
    titles: list[str],
    coords_2d: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    model: str = "gemma2:9b",
    n_samples: int = 20,
) -> list[str]:
    """Label flat clusters by sampling from their 2D spatial extent.

    Instead of inheriting labels from dendrogram nodes, this function
    directly labels each cluster at a given cut level by:
    1. Sampling titles from the cluster's 2D spatial extent (not embedding centrality)
    2. Computing contrastive TF-IDF keywords against the spatially nearest cluster
    3. Asking the LLM to produce a topic label

    Args:
        titles: List of all titles
        coords_2d: 2D coordinates for all points
        labels: Cluster assignment for each point (from fcluster)
        centroids: 2D centroids for each cluster
        n_clusters: Number of clusters
        model: Ollama model name
        n_samples: Number of sample titles per cluster

    Returns:
        List of cluster names (one per cluster)
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"  Labeling {n_clusters} clusters (post-hoc, spatial sampling)...")

    # Group point indices by cluster
    cluster_points = defaultdict(list)
    for i, cid in enumerate(labels):
        cluster_points[int(cid)].append(i)

    def _sample_spatial(point_indices, k):
        """Sample points spread across the 2D spatial extent of a cluster."""
        pts = np.array(point_indices)
        if len(pts) <= k:
            return pts.tolist()

        cluster_coords = coords_2d[pts]
        # Farthest-point sampling in 2D for spatial coverage
        chosen = [np.random.randint(len(pts))]  # random seed point
        for _ in range(k - 1):
            chosen_coords = cluster_coords[chosen]
            # Distance from each candidate to nearest chosen point
            dists = np.min(
                np.linalg.norm(
                    cluster_coords[:, None, :] - chosen_coords[None, :, :],
                    axis=2
                ),
                axis=1
            )
            dists[chosen] = -1  # exclude already chosen
            chosen.append(int(np.argmax(dists)))

        return pts[chosen].tolist()

    # Build all prompts
    label_tasks = []  # (cluster_id, prompt)
    for cluster_id in range(n_clusters):
        pts = cluster_points[cluster_id]
        if not pts:
            continue

        # Sample from 2D spatial extent, then deduplicate titles
        sample_indices = _sample_spatial(pts, n_samples * 3)  # oversample to survive dedup
        seen = set()
        sample_titles = []
        for i in sample_indices:
            t = titles[i]
            if t not in seen:
                seen.add(t)
                sample_titles.append(t)
                if len(sample_titles) >= n_samples:
                    break

        # Contrastive TF-IDF against nearest spatial neighbor
        nearest_id = find_nearest_cluster(cluster_id, centroids)
        neighbor_pts = cluster_points[nearest_id]

        kw_str = ""
        if neighbor_pts:
            combined_titles = [titles[p] for p in pts] + [titles[p] for p in neighbor_pts]
            combined_labels_arr = np.zeros(len(pts) + len(neighbor_pts), dtype=int)
            combined_labels_arr[len(pts):] = 1
            kw = compute_tfidf_keywords(combined_titles, combined_labels_arr, 2, top_k=8, min_df=1)
            keywords = [w for w, _ in kw.get(0, [])][:8]
            if keywords:
                kw_str = f"\nDistinguishing keywords (vs neighbor): {', '.join(keywords)}"

        prompt = f"""You are labeling clusters on a Wikipedia article map. This cluster has {len(pts)} articles.
{kw_str}
Sample articles from across this cluster:
{chr(10).join(f'- {t}' for t in sample_titles)}

Give a short (2-5 word) label for this cluster. The label should name the SPECIFIC subject area, not a vague category.

BAD labels (too vague): "Human Knowledge", "Historical Events", "Human Culture", "Various Topics", "Human Civilization"
GOOD labels: "Anatomy & Medicine", "Cold War Politics", "European Monarchs", "Programming Languages", "Olympic Sports", "African Geography"

Reply with ONLY the label, nothing else."""

        label_tasks.append((cluster_id, prompt))

    # Call LLM in parallel
    cluster_names = [f"Cluster {i}" for i in range(n_clusters)]

    def _call_ollama(task):
        node_id, prompt = task
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            label = result.stdout.strip().split('\n')[0][:50]
            label = label.strip('"\'').strip()
            return node_id, label if label else f"Cluster {node_id}"
        except Exception:
            return node_id, f"Cluster {node_id}"

    n_workers = min(4, len(label_tasks))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_call_ollama, task): task for task in label_tasks}
        for future in as_completed(futures):
            cid, label = future.result()
            cluster_names[cid] = label
            completed += 1
            if completed % 10 == 0:
                print(f"    Labeled {completed}/{len(label_tasks)} clusters...", flush=True)

    print(f"  Done labeling {n_clusters} clusters.")
    return cluster_names


def label_clusters_contrastive(
    titles: list[str],
    labels: np.ndarray,
    n_clusters: int,
    centroids: np.ndarray,
    model: str = "gemma2:9b",
) -> list[str]:
    """Generate contrastive cluster labels using TF-IDF keywords + LLM.

    For each cluster:
    1. Compute TF-IDF keywords that distinguish it
    2. Find nearest neighbor cluster
    3. Ask LLM to generate contrastive label
    """
    print("  Computing TF-IDF keywords...")
    keywords = compute_tfidf_keywords(titles, labels, n_clusters)

    cluster_names = []

    # Group titles by cluster for sampling
    from collections import defaultdict
    cluster_titles = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_titles[int(label)].append(titles[i])

    for cluster_id in range(n_clusters):
        this_titles = cluster_titles[cluster_id]

        if not this_titles:
            cluster_names.append(f"Cluster {cluster_id}")
            continue

        # Get keywords for this cluster
        this_keywords = [w for w, _ in keywords.get(cluster_id, [])][:5]

        # Find nearest cluster for contrast
        nearest_id = find_nearest_cluster(cluster_id, centroids)
        neighbor_titles = cluster_titles[nearest_id]
        neighbor_keywords = [w for w, _ in keywords.get(nearest_id, [])][:5]

        # Sample titles
        sample_this = this_titles[:10] if len(this_titles) <= 10 else \
                      [this_titles[i] for i in np.random.choice(len(this_titles), 10, replace=False)]
        sample_neighbor = neighbor_titles[:10] if len(neighbor_titles) <= 10 else \
                          [neighbor_titles[i] for i in np.random.choice(len(neighbor_titles), 10, replace=False)]

        prompt = f"""Two clusters of Wikipedia articles need distinct labels.

CLUSTER A (label this one):
Keywords: {', '.join(this_keywords) if this_keywords else 'none'}
Sample titles: {', '.join(sample_this)}

CLUSTER B (nearby, for contrast):
Keywords: {', '.join(neighbor_keywords) if neighbor_keywords else 'none'}
Sample titles: {', '.join(sample_neighbor)}

Give a 2-3 word label for CLUSTER A that distinguishes it from CLUSTER B.
Reply with ONLY the label, nothing else."""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            label = result.stdout.strip().split('\n')[0][:25]
            cluster_names.append(label if label else f"Cluster {cluster_id}")
            print(f"  Cluster {cluster_id}: {cluster_names[-1]} (keywords: {', '.join(this_keywords[:3])})")
        except Exception as e:
            print(f"  Cluster {cluster_id}: LLM error - {e}")
            cluster_names.append(f"Cluster {cluster_id}")

    return cluster_names


def label_clusters_with_llm(
    titles: list[str],
    labels: np.ndarray,
    n_clusters: int,
    model: str = "gemma2:9b",
    centroids: np.ndarray | None = None,
    contrastive: bool = True,
) -> list[str]:
    """Generate cluster labels using local Ollama.

    Args:
        titles: List of all titles
        labels: Cluster assignment for each title
        n_clusters: Number of clusters
        model: Ollama model to use
        centroids: Cluster centroids (required if contrastive=True)
        contrastive: Use contrastive labeling with TF-IDF keywords
    """
    if contrastive and centroids is not None:
        return label_clusters_contrastive(titles, labels, n_clusters, centroids, model)

    # Fallback to simple labeling
    cluster_names = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_titles = [titles[i] for i in range(len(titles)) if mask[i]]

        if not cluster_titles:
            cluster_names.append(f"Cluster {cluster_id}")
            continue

        sample = cluster_titles[:20] if len(cluster_titles) <= 20 else \
                 [cluster_titles[i] for i in np.random.choice(len(cluster_titles), 20, replace=False)]

        prompt = f"""These are article titles from a cluster. Give a 2-3 word label describing their common theme.

Titles: {', '.join(sample)}

Reply with ONLY the label, nothing else."""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            label = result.stdout.strip().split('\n')[0][:20]
            cluster_names.append(label if label else f"Cluster {cluster_id}")
            print(f"  Cluster {cluster_id}: {cluster_names[-1]}")
        except Exception as e:
            print(f"  Cluster {cluster_id}: LLM error - {e}")
            cluster_names.append(f"Cluster {cluster_id}")

    return cluster_names


def compute_lsh_visualization(coords_2d: np.ndarray, embeddings: np.ndarray, num_bits: int = 12):
    """Compute LSH bucket assignments and hyperplane projections for visualization.

    Args:
        coords_2d: UMAP 2D coordinates
        embeddings: Original high-dim embeddings
        num_bits: Number of LSH bits (determines number of hyperplanes)

    Returns:
        dict with:
        - 'bucket_ids': LSH bucket ID per point
        - 'hyperplanes_2d': Projected hyperplanes as line segments in 2D
        - 'bucket_centroids': Centroid of each bucket in 2D
        - 'bucket_sizes': Number of points per bucket
        - 'boundary_points': Points near bucket boundaries
    """
    from sklearn.decomposition import PCA
    from collections import defaultdict

    print(f"Computing LSH visualization ({num_bits} bits)...")
    n, d = embeddings.shape

    # Step 1: Random hyperplanes -> initial buckets
    rng = np.random.default_rng(42)
    random_hp = rng.standard_normal((num_bits, d)).astype(np.float32)
    random_hp = random_hp / np.linalg.norm(random_hp, axis=1, keepdims=True)

    signs_random = (embeddings @ random_hp.T) >= 0
    powers = 2 ** np.arange(num_bits)
    hashes_random = (signs_random @ powers).astype(np.uint64)

    # Compute random bucket centroids
    random_bucket_to_indices = defaultdict(list)
    for idx, h in enumerate(hashes_random):
        random_bucket_to_indices[int(h)].append(idx)

    centroids = []
    for bid, indices in random_bucket_to_indices.items():
        if len(indices) >= 2:
            centroid = embeddings[indices].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroids.append(centroid / norm)

    centroids = np.array(centroids, dtype=np.float32)
    print(f"  Random hash: {len(random_bucket_to_indices):,} buckets, {len(centroids)} centroids")

    # Step 2: PCA on centroids -> data-adapted hyperplanes
    n_components = min(num_bits, len(centroids) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(centroids)
    hp = pca.components_.astype(np.float32)  # Shape: (n_components, d)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Step 3: Re-hash with PCA hyperplanes
    signs = (embeddings @ hp.T) >= 0
    bucket_ids = (signs @ powers[:len(hp)]).astype(np.uint64)

    # Step 4: Compute bucket centroids in 2D
    bucket_to_indices = defaultdict(list)
    for idx, h in enumerate(bucket_ids):
        bucket_to_indices[int(h)].append(idx)

    bucket_centroids_2d = {}
    bucket_sizes = {}
    for bid, indices in bucket_to_indices.items():
        bucket_sizes[bid] = len(indices)
        if indices:
            bucket_centroids_2d[bid] = coords_2d[indices].mean(axis=0)

    print(f"  Final: {len(bucket_to_indices):,} buckets")

    # Step 5: Project hyperplanes to 2D for visualization
    # Use linear regression to find best 2D representation of hyperplane cuts
    # For each hyperplane, find points just above/below threshold and draw separator
    hyperplanes_2d = []
    for hp_idx in range(len(hp)):
        # Get projections onto this hyperplane
        projections = embeddings @ hp[hp_idx]

        # Find points near the boundary (within 10% of range around 0)
        proj_range = projections.max() - projections.min()
        near_boundary = np.abs(projections) < 0.1 * proj_range

        if near_boundary.sum() > 10:
            # Fit a line through boundary points in 2D
            boundary_coords = coords_2d[near_boundary]
            # Use PCA to find principal direction of boundary points
            if len(boundary_coords) > 2:
                pca_2d = PCA(n_components=1)
                pca_2d.fit(boundary_coords)
                direction = pca_2d.components_[0]
                center = boundary_coords.mean(axis=0)

                # Extend line across the plot
                extent = 20  # How far to extend
                p1 = center - direction * extent
                p2 = center + direction * extent
                hyperplanes_2d.append({
                    'x': [float(p1[0]), float(p2[0])],
                    'y': [float(p1[1]), float(p2[1])],
                    'bit': hp_idx,
                })

    print(f"  Projected {len(hyperplanes_2d)} hyperplanes to 2D")

    # Step 6: Find boundary points (points with low margin to any hyperplane)
    margins = np.abs(embeddings @ hp.T)
    min_margins = margins.min(axis=1)
    margin_threshold = np.percentile(min_margins, 20)  # Bottom 20%
    boundary_points = np.where(min_margins < margin_threshold)[0]
    print(f"  Found {len(boundary_points)} boundary points")

    return {
        'bucket_ids': bucket_ids,
        'hyperplanes_2d': hyperplanes_2d,
        'bucket_centroids_2d': bucket_centroids_2d,
        'bucket_sizes': bucket_sizes,
        'boundary_points': boundary_points.tolist(),
        'num_bits': num_bits,
        'bucket_to_indices': dict(bucket_to_indices),
    }


def compute_lsh_hierarchy(
    coords_2d: np.ndarray,
    titles: list[str],
    embeddings: np.ndarray,
    n_micro_clusters: int = 1000,
) -> dict:
    """Build dendrogram hierarchy using k-means micro-clusters on 2D coords.

    K-means on 2D UMAP coordinates produces spatially contiguous micro-clusters.
    Ward linkage on their 2D centroids builds the hierarchy. Node labels are
    cheap placeholders; real labels come from post-hoc label_flat_clusters().

    Returns the same dict contract as compute_dendrogram_hierarchy() for
    panel compatibility.
    """
    from scipy.cluster.hierarchy import to_tree
    from collections import defaultdict

    print("Computing spatial hierarchy...")

    # K-means on 2D coords for spatially contiguous micro-clusters
    n_micro = min(n_micro_clusters, len(coords_2d) // 10)
    print(f"  K-means with k={n_micro} on 2D coordinates...")
    kmeans = KMeans(n_clusters=n_micro, random_state=42, n_init=10)
    kmeans.fit(coords_2d)

    point_to_micro = kmeans.labels_

    # Ward linkage on 2D micro-cluster centroids
    print("  Building dendrogram (Ward linkage on 2D centroids)...")
    Z = linkage(kmeans.cluster_centers_, method='ward')

    # Build tree and collect node info
    tree = to_tree(Z, rd=True)
    root, nodes = tree

    micro_to_points = defaultdict(list)
    for i, micro_id in enumerate(point_to_micro):
        micro_to_points[micro_id].append(i)

    node_points = {}
    node_centroids = {}

    def get_node_points(node):
        if node.is_leaf():
            return micro_to_points[node.id]
        return get_node_points(node.left) + get_node_points(node.right)

    print("  Collecting node info...")
    for node in nodes:
        points = get_node_points(node)
        node_points[node.id] = points
        if points:
            centroid = coords_2d[points].mean(axis=0)
            node_centroids[node.id] = (float(centroid[0]), float(centroid[1]))
        else:
            node_centroids[node.id] = (0.0, 0.0)

    # Build parent map from Z for label inheritance
    parent_map = {}
    for i, row in enumerate(Z):
        node_id = n_micro + i
        parent_map[int(row[0])] = node_id
        parent_map[int(row[1])] = node_id

    # Label leaf nodes cheaply (representative title, no LLM)
    node_labels = {}
    for node in nodes:
        if node.is_leaf():
            pts = node_points[node.id]
            if pts:
                # Use embedding-based representative if enough points
                if len(pts) > 3:
                    sample_pts = sample_representative_points(pts, embeddings, n_samples=3, method="central")
                else:
                    sample_pts = pts[:3]
                node_labels[node.id] = titles[sample_pts[0]][:25]
            else:
                node_labels[node.id] = f"Bucket {node.id}"

    # Label internal nodes cheaply (for panel's dynamic cutting fallback).
    # Real labels come from post-hoc label_flat_clusters() on the cut results.
    for node in nodes:
        if not node.is_leaf() and node.id not in node_labels:
            pts = node_points[node.id]
            if pts:
                # Use the title of a spatially central point as placeholder
                cluster_coords = coords_2d[pts]
                centroid = cluster_coords.mean(axis=0)
                dists = np.linalg.norm(cluster_coords - centroid, axis=1)
                node_labels[node.id] = titles[pts[int(np.argmin(dists))]][:25]
            else:
                node_labels[node.id] = f"Node {node.id}"

    print("  Tree labels assigned (placeholders for dynamic cutting).")

    max_dist = float(Z[:, 2].max()) if len(Z) > 0 else 1.0

    # Compute 2D centroids for each micro-cluster (for label placement)
    kmeans_centroids_2d = np.zeros((n_micro, 2))
    for leaf_idx in range(n_micro):
        pts = micro_to_points[leaf_idx]
        if pts:
            kmeans_centroids_2d[leaf_idx] = coords_2d[pts].mean(axis=0)

    return {
        'Z': Z,
        'kmeans_labels': point_to_micro,           # compat: point -> micro-cluster index
        'kmeans_centroids': kmeans_centroids_2d,    # compat: 2D centroids for label placement
        'node_labels': node_labels,
        'node_points': {k: v for k, v in node_points.items()},
        'node_centroids': node_centroids,
        'n_micro': n_micro,
        'max_dist': max_dist,
    }


def compute_bridge_edges(coords_2d: np.ndarray, embeddings: np.ndarray, cluster_labels: dict,
                         bridge_cluster_level: int = 12):
    """Compute bridge edges using dyf ROG ontology.

    Aggregates individual cross-cluster connections into cluster-to-cluster edges.
    This shows which knowledge regions connect without overwhelming detail.

    Args:
        coords_2d: UMAP coordinates for bundling
        embeddings: Original embeddings for ROG
        cluster_labels: Dict mapping cluster level -> array of labels per point
        bridge_cluster_level: Which cluster level to use for bridge detection (default: 12)
    """
    from collections import defaultdict

    print("Building ROG ontology for bridge detection...")

    # Build ROG ontology
    result = dyf.build_rog_ontology(
        embeddings,
        initial_threshold=0.55,
        min_threshold=0.35,
        target_coverage=0.95,
        verbose=True,
    )

    # Get cluster labels at the bridge detection level
    labels = cluster_labels[bridge_cluster_level]
    print(f"Using {bridge_cluster_level}-cluster level for bridge detection")

    # Count connections between cluster pairs
    ont = result.ontology
    cluster_pair_counts = defaultdict(int)
    cluster_pair_points = defaultdict(list)  # Track actual point pairs for highlighting
    within_cluster = 0
    cross_cluster = 0

    for parent, children_list in ont.children.items():
        for child, sim, div_gap in children_list:
            c1, c2 = labels[parent], labels[child]
            if c1 != c2:
                cross_cluster += 1
                # Normalize pair order
                pair = (min(c1, c2), max(c1, c2))
                cluster_pair_counts[pair] += 1
                # Store point pair for highlighting
                if parent < child:
                    cluster_pair_points[pair].append((parent, child))
                else:
                    cluster_pair_points[pair].append((child, parent))
            else:
                within_cluster += 1

    total_edges = within_cluster + cross_cluster
    print(f"Edge analysis: {within_cluster:,} within-cluster, {cross_cluster:,} cross-cluster")
    print(f"Found {len(cluster_pair_counts)} unique cluster pairs with connections")

    if not cluster_pair_counts:
        print("No bridge edges found")
        return {level: ([], []) for level in cluster_labels.keys()}, [], {}, result.ontology.diversity

    # Compute cluster centroids
    n_clusters = bridge_cluster_level
    centroids = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids.append(coords_2d[mask].mean(axis=0))
        else:
            centroids.append(np.array([0.0, 0.0]))
    centroids = np.array(centroids)

    # Create edges between cluster centroids
    # We'll create multiple edges for pairs with more connections (visual weight)
    print("Creating cluster-to-cluster edges...")
    cluster_edges = []
    for (c1, c2), count in sorted(cluster_pair_counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {c1} <-> Cluster {c2}: {count:,} connections")
        cluster_edges.append((c1, c2))

    # Bundle edges using centroid positions
    print(f"Bundling {len(cluster_edges)} cluster-to-cluster edges...")
    nodes = pd.DataFrame({'x': centroids[:, 0], 'y': centroids[:, 1]})
    nodes.index.name = 'id'

    edges_df = pd.DataFrame(cluster_edges, columns=['source', 'target'])
    bundled = hammer_bundle(
        nodes, edges_df,
        iterations=BUNDLE_ITERATIONS,
        batch_size=min(20000, len(cluster_edges)),
    )

    # Convert to multi_line format
    print("Converting to multi_line format...")
    xs, ys = _bundled_to_multiline(bundled)
    print(f"Converted to {len(xs):,} edge paths")

    # Return same processed edges for all levels
    edges_result = {level: (xs, ys) for level in cluster_labels.keys()}

    # Flatten all point pairs for highlighting (edge_indices)
    edge_indices = []
    for pair_list in cluster_pair_points.values():
        edge_indices.extend(pair_list)

    # Return cluster pair info for potential use (edge weights, etc.)
    cluster_pair_info = dict(cluster_pair_counts)

    return edges_result, edge_indices, cluster_pair_info, result.ontology.diversity


def _bundled_to_multiline(bundled: pd.DataFrame) -> tuple[list, list]:
    """Convert hammer_bundle DataFrame to multi_line format (xs, ys lists)."""
    if bundled.empty:
        return [], []

    # Faster vectorized approach: find NaN boundaries and split
    x_vals = bundled['x'].values
    y_vals = bundled['y'].values
    nan_mask = np.isnan(x_vals) | np.isnan(y_vals)

    # Find indices where NaN occurs (edge boundaries)
    nan_indices = np.where(nan_mask)[0]

    xs, ys = [], []
    start = 0

    for end in nan_indices:
        if end > start:
            xs.append(x_vals[start:end].tolist())
            ys.append(y_vals[start:end].tolist())
        start = end + 1

    # Handle final segment
    if start < len(x_vals):
        xs.append(x_vals[start:].tolist())
        ys.append(y_vals[start:].tolist())

    return xs, ys


def cut_dendrogram(dendrogram: dict, n_clusters: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cut dendrogram to get n_clusters and return labels, centroids, and names.

    Args:
        dendrogram: Output from compute_dendrogram_hierarchy()
        n_clusters: Number of clusters to cut to

    Returns:
        labels: Array of cluster assignments for each point
        centroids: Array of cluster centroids (n_clusters, 2)
        names: List of cluster names
    """
    Z = dendrogram['Z']
    kmeans_labels = dendrogram['kmeans_labels']
    node_labels = dendrogram['node_labels']
    node_points = dendrogram['node_points']
    node_centroids = dendrogram['node_centroids']
    n_micro = dendrogram['n_micro']

    # Cut dendrogram at n_clusters
    micro_cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

    # Map point labels through kmeans -> dendrogram cut
    point_labels = np.array([micro_cluster_labels[m] for m in kmeans_labels])

    # Find representative internal nodes for each cut cluster
    # These are the nodes that became cluster roots at this cut level
    from scipy.cluster.hierarchy import to_tree

    tree = to_tree(Z, rd=True)
    root, nodes = tree

    # Find which internal nodes correspond to each cluster at this cut
    # We traverse the tree and find nodes where all descendants belong to same cluster
    cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
    cluster_centroids = []

    for cluster_id in range(n_clusters):
        # Find points in this cluster
        mask = point_labels == cluster_id
        cluster_point_indices = np.where(mask)[0]

        if len(cluster_point_indices) == 0:
            cluster_centroids.append(np.array([0.0, 0.0]))
            continue

        # Find the internal node that best represents this cluster
        # (the deepest node that contains all cluster points and only cluster points)
        best_node = None
        best_size = float('inf')

        for node in nodes:
            if node.is_leaf():
                continue
            node_pts = set(node_points[node.id])
            cluster_pts = set(cluster_point_indices)

            # Check if this node contains the cluster
            if cluster_pts.issubset(node_pts):
                if len(node_pts) < best_size:
                    best_size = len(node_pts)
                    best_node = node

        if best_node is not None and best_node.id in node_labels:
            cluster_names[cluster_id] = node_labels[best_node.id]

        # Compute centroid from actual points
        centroid = dendrogram['kmeans_centroids'][kmeans_labels[cluster_point_indices[0]]]
        # Actually use the node centroid if available, or compute from points
        if best_node is not None:
            cx, cy = node_centroids[best_node.id]
            cluster_centroids.append(np.array([cx, cy]))
        else:
            # Fallback: average of micro-cluster centroids in this cluster
            micro_ids = np.unique(kmeans_labels[cluster_point_indices])
            centroid = dendrogram['kmeans_centroids'][micro_ids].mean(axis=0)
            cluster_centroids.append(centroid)

    return point_labels, np.array(cluster_centroids), cluster_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", default="demo/wiki_simple_50k.parquet")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--bridge-level", type=int, default=100,
                        help="Cluster level for bridge detection (12, 25, 50, or 100)")
    parser.add_argument("--use-dendrogram", action="store_true",
                        help="Use dynamic dendrogram hierarchy (labels all internal nodes)")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--densmap", action="store_true",
                        help="Use DENSMAP for density-preserving projection")
    args = parser.parse_args()

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.data_path)
        output_path = input_path.parent / f"{input_path.stem}_rog_cache.pkl"

    print(f"Will save to: {output_path}")

    # Process
    coords_2d, titles, embeddings = load_data(
        args.data_path, args.sample,
        n_neighbors=args.umap_neighbors, densmap=args.densmap)

    # Compute LSH visualization data first (needed for LSH-based hierarchy)
    lsh_data = compute_lsh_visualization(coords_2d, embeddings, num_bits=12)

    if args.use_dendrogram:
        # LSH-based hierarchy: uses LSH buckets as micro-clusters
        print("\n=== Using LSH-based dendrogram hierarchy ===")
        dendrogram = compute_lsh_hierarchy(coords_2d, titles, embeddings)

        # Pre-compute cluster results for standard levels
        cluster_result = {'labels': {}, 'names': {}, 'centroids': {}}
        for n_clusters in CLUSTER_LEVELS:
            print(f"Cutting dendrogram at {n_clusters} clusters...")
            labels, centroids, names = cut_dendrogram(dendrogram, n_clusters)
            cluster_result['labels'][n_clusters] = labels
            cluster_result['centroids'][n_clusters] = centroids
            cluster_result['names'][n_clusters] = names  # placeholder

        # Post-hoc labeling: label flat clusters directly from their spatial contents
        print("\n=== Post-hoc cluster labeling ===")
        for n_clusters in CLUSTER_LEVELS:
            labels = cluster_result['labels'][n_clusters]
            centroids = cluster_result['centroids'][n_clusters]
            names = label_flat_clusters(
                titles, coords_2d, labels, centroids, n_clusters)
            names = disambiguate_cluster_names(names, titles, labels)
            cluster_result['names'][n_clusters] = names

        # Add dendrogram to cluster_result for dynamic cutting
        cluster_result['dendrogram'] = dendrogram
    else:
        # Legacy fixed-level approach
        cluster_result = compute_hierarchical_clusters(coords_2d, titles)
        dendrogram = None

    bridge_edges, edge_indices, cluster_pairs, diversity = compute_bridge_edges(
        coords_2d, embeddings, cluster_result['labels'],
        bridge_cluster_level=args.bridge_level
    )

    # Compute multi-resolution analysis using Rust DensityClassifier
    from dyf_rs import DensityClassifier as RustClassifier
    num_bits = 12
    rust_clf = RustClassifier(embedding_dim=embeddings.shape[1], num_bits=num_bits, seed=42)
    rust_clf.fit(embeddings.astype(np.float32))
    lsh_data['rust_bucket_ids'] = rust_clf.get_bucket_ids()

    # Multi-resolution analysis (may not be available in all dyf_rs versions)
    if hasattr(rust_clf, 'multi_resolution_analysis'):
        print("Computing multi-resolution analysis...")
        mra = rust_clf.multi_resolution_analysis(dense_threshold=10)
        lsh_data['recovery_depth'] = mra.recovery_depth
        lsh_data['recovery_ratio'] = mra.recovery_ratio
        lsh_data['buckets_per_depth'] = mra.buckets_per_depth
        lsh_data['mean_size_per_depth'] = mra.mean_size_per_depth
        lsh_data['mra_dense_threshold'] = mra.dense_threshold
        print(f"  Multi-resolution: {sum(1 for d in mra.recovery_depth if d == 0)} already dense, "
              f"{sum(1 for d in mra.recovery_depth if 0 < d <= num_bits)} recovered, "
              f"{sum(1 for d in mra.recovery_depth if d > num_bits)} never recovered")
    else:
        print("Skipping multi-resolution analysis (not available in this dyf_rs version)")

    # Bridge persistence analysis (may not be available in all dyf_rs versions)
    if hasattr(rust_clf, 'bridge_persistence'):
        print("Computing bridge persistence analysis...")
        bp = rust_clf.bridge_persistence(embeddings.astype(np.float32), relative_threshold=0.8)
        lsh_data['bridge_persistence'] = bp.bridge_persistence
        lsh_data['max_bridge_depth'] = bp.max_bridge_depth
        lsh_data['min_bridge_depth'] = bp.min_bridge_depth
        lsh_data['bridge_ratio'] = bp.bridge_ratio
        lsh_data['bridges_per_depth'] = bp.bridges_per_depth
        total_connectors = sum(1 for p in bp.bridge_persistence if p > 0)
        max_p = max(bp.bridge_persistence) if bp.bridge_persistence else 0
        print(f"  Connectors: {total_connectors}, max persistence={max_p}, threshold={bp.relative_threshold}")
        print(f"  Connectors per depth: {bp.bridges_per_depth}")
    else:
        print("Skipping bridge persistence analysis (not available in this dyf_rs version)")

    # Save
    cache = {
        'coords_2d': coords_2d,
        'titles': titles,
        'cluster_result': cluster_result,
        'bridge_edges': bridge_edges,
        'edge_indices': edge_indices,  # (source, target) pairs for highlighting
        'cluster_pairs': cluster_pairs,  # {(c1, c2): count} for edge weights
        'diversity': diversity,  # For visualization (ROG diversity scores)
        'lsh_data': lsh_data,  # LSH bucket visualization data
    }

    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"\nSaved cache to {output_path}")
    print(f"Run server with: bokeh serve demo/rog_panel.py --port 5007 --args {output_path}")
    if args.use_dendrogram:
        print("Note: Dendrogram data included - panel can dynamically cut at any level")
    print("Note: LSH data included - toggle LSH mode to see bucket assignments")


if __name__ == "__main__":
    main()
