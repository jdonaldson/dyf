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
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
import umap
from datashader.bundling import hammer_bundle
import subprocess

CLUSTER_LEVELS = (8, 15, 30)
EDGE_K = 5
BUNDLE_ITERATIONS = 4


def load_data(path: str, sample: int | None = None):
    """Load parquet and run UMAP."""
    print(f"Loading {path}...")
    df = pl.read_parquet(path)

    if sample and sample < len(df):
        df = df.sample(sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list())

    print(f"Running UMAP on {len(titles)} points...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(embeddings)

    return coords_2d, titles


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


def compute_bridge_edges(coords_2d: np.ndarray, cluster_labels: dict):
    """Compute bridge edges for each cluster level."""
    print(f"Computing {EDGE_K}-NN graph...")

    nn = NearestNeighbors(n_neighbors=EDGE_K + 1, algorithm='auto')
    nn.fit(coords_2d)
    _, indices = nn.kneighbors(coords_2d)

    all_edges = []
    for i in range(len(coords_2d)):
        for j in indices[i, 1:]:
            if i < j:
                all_edges.append((i, j))

    print(f"Found {len(all_edges)} total k-NN edges")

    nodes = pd.DataFrame({'x': coords_2d[:, 0], 'y': coords_2d[:, 1]})
    nodes.index.name = 'id'

    result = {}
    for level, labels in cluster_labels.items():
        bridge_edges = [(i, j) for i, j in all_edges if labels[i] != labels[j]]
        print(f"Level {level}: {len(bridge_edges)} bridge edges")

        if not bridge_edges:
            result[level] = pd.DataFrame({'x': [], 'y': []})
            continue

        edges_df = pd.DataFrame(bridge_edges, columns=['source', 'target'])
        bundled = hammer_bundle(
            nodes, edges_df,
            iterations=BUNDLE_ITERATIONS,
            batch_size=min(20000, len(bridge_edges)),
        )
        result[level] = bundled

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", default="demo/wiki_simple_50k.parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.data_path)
        output_path = input_path.parent / f"{input_path.stem}_rog_cache.pkl"

    print(f"Will save to: {output_path}")

    # Process
    coords_2d, titles = load_data(args.data_path, args.sample)
    cluster_result = compute_hierarchical_clusters(coords_2d, titles)
    bridge_edges = compute_bridge_edges(coords_2d, cluster_result['labels'])

    # Save
    cache = {
        'coords_2d': coords_2d,
        'titles': titles,
        'cluster_result': cluster_result,
        'bridge_edges': bridge_edges,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"\nSaved cache to {output_path}")
    print(f"Run server with: bokeh serve demo/rog_panel.py --port 5007 --args {output_path}")


if __name__ == "__main__":
    main()
