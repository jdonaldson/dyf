"""
Preprocess wiki data for ROG Browser.
Generates UMAP coords, clusters, LLM labels, and bundled edges.

Run: python demo/rog_preprocess.py demo/wiki_simple_50k.parquet --sample 10000
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

CLUSTER_LEVELS = (5, 12, 25, 50)
BUNDLE_ITERATIONS = 4


def load_data(path: str, sample: int | None = None):
    """Load parquet and run UMAP. Returns coords, titles, and embeddings."""
    print(f"Loading {path}...")
    df = pl.read_parquet(path)

    if sample and sample < len(df):
        df = df.sample(sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list())

    print(f"Running UMAP on {len(titles)} points (parallel)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1)
    coords_2d = reducer.fit_transform(embeddings)

    return coords_2d, titles, embeddings


def compute_hierarchical_clusters(coords_2d: np.ndarray, titles: list[str]):
    """Cluster using k-means then cut dendrogram at multiple levels."""
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

    # LLM label all levels
    for n_clusters in CLUSTER_LEVELS:
        print(f"Generating LLM labels for {n_clusters} clusters...")
        result['names'][n_clusters] = label_clusters_with_llm(
            titles, result['labels'][n_clusters], n_clusters
        )

    return result


def label_clusters_with_llm(
    titles: list[str],
    labels: np.ndarray,
    n_clusters: int,
    model: str = "gemma2:9b",
) -> list[str]:
    """Generate cluster labels using local Ollama."""
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


def compute_bridge_edges(coords_2d: np.ndarray, embeddings: np.ndarray, cluster_labels: dict,
                         bridge_cluster_level: int = 5):
    """Compute bridge edges using dyf ROG ontology.

    Aggregates individual cross-cluster connections into cluster-to-cluster edges.
    This shows which knowledge regions connect without overwhelming detail.

    Args:
        coords_2d: UMAP coordinates for bundling
        embeddings: Original embeddings for ROG
        cluster_labels: Dict mapping cluster level -> array of labels per point
        bridge_cluster_level: Which cluster level to use for bridge detection (default: coarsest)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", default="demo/wiki_simple_50k.parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--bridge-level", type=int, default=50,
                        help="Cluster level for bridge detection (5, 12, 25, or 50)")
    args = parser.parse_args()

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.data_path)
        output_path = input_path.parent / f"{input_path.stem}_rog_cache.pkl"

    print(f"Will save to: {output_path}")

    # Process
    coords_2d, titles, embeddings = load_data(args.data_path, args.sample)
    cluster_result = compute_hierarchical_clusters(coords_2d, titles)
    bridge_edges, edge_indices, cluster_pairs, diversity = compute_bridge_edges(
        coords_2d, embeddings, cluster_result['labels'],
        bridge_cluster_level=args.bridge_level
    )

    # Save
    cache = {
        'coords_2d': coords_2d,
        'titles': titles,
        'cluster_result': cluster_result,
        'bridge_edges': bridge_edges,
        'edge_indices': edge_indices,  # (source, target) pairs for highlighting
        'cluster_pairs': cluster_pairs,  # {(c1, c2): count} for edge weights
        'diversity': diversity,  # For visualization (ROG diversity scores)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"\nSaved cache to {output_path}")
    print(f"Run server with: bokeh serve demo/rog_panel.py --port 5007 --args {output_path}")


if __name__ == "__main__":
    main()
