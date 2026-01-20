"""
Wikipedia Knowledge Graph Visualization

Generates an interactive visualization of Wikipedia article embeddings,
showing clusters and density-based bridges between them.

Requirements:
    pip install dyf umap-learn plotly scikit-learn requests

Usage:
    python wiki_visualization.py embeddings.parquet --output wiki_graph.html
    python wiki_visualization.py embeddings.parquet --label-with-ollama
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    import requests
except ImportError:
    requests = None

from dyf import DensityClassifier, BridgeAnalysis


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str], Optional[List[str]]]:
    """Load embeddings from parquet file.

    Expected columns: 'embedding', 'title' or 'text', optionally 'category'
    """
    if pl is None:
        raise ImportError("polars required: pip install polars")

    df = pl.read_parquet(path)

    embeddings = np.array(df['embedding'].to_list(), dtype=np.float32)

    # Get titles/text
    if 'title' in df.columns:
        texts = df['title'].to_list()
    elif 'text' in df.columns:
        texts = df['text'].to_list()
    else:
        texts = [f"Item {i}" for i in range(len(embeddings))]

    # Get categories if available
    categories = df['category'].to_list() if 'category' in df.columns else None

    return embeddings, texts, categories


def project_to_2d(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Project embeddings to 2D using UMAP."""
    if UMAP is None:
        raise ImportError("umap-learn required: pip install umap-learn")

    print("Projecting to 2D with UMAP...")
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d


def compute_edge_bundles(
    coords_2d: np.ndarray,
    bridge_analysis: BridgeAnalysis,
    classifier: DensityClassifier,
    n_top_pairs: int = 300,
    max_edges_per_pair: int = 5
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """Compute edge-bundled paths between bridge points using datashader hammer_bundle.

    Draws edges between actual bridge points in connected buckets, not just centroids.

    Returns (medium_x, medium_y, heavy_x, heavy_y) for Plotly line traces.
    """
    import pandas as pd
    from datashader.bundling import hammer_bundle

    bucket_ids = classifier.get_bucket_ids()
    bridge_indices = set(bridge_analysis.bridge_indices)

    # Group bridge points by bucket
    bucket_to_bridges = defaultdict(list)
    for idx in bridge_indices:
        bucket_to_bridges[bucket_ids[idx]].append(idx)

    # Get bridge connections between buckets
    top_pairs = bridge_analysis.top_connected_pairs(n_top_pairs)

    # Use 75th percentile as threshold for heavy edges
    counts = [count for _, _, count in top_pairs]
    heavy_threshold = np.percentile(counts, 75) if counts else 0

    # Build node list from all bridge points
    all_bridge_indices = sorted(bridge_indices)
    idx_to_node = {idx: i for i, idx in enumerate(all_bridge_indices)}
    nodes_data = [{'x': coords_2d[idx, 0], 'y': coords_2d[idx, 1]} for idx in all_bridge_indices]
    nodes_df = pd.DataFrame(nodes_data)

    # Build edges between bridge points in connected buckets
    medium_edges = []
    heavy_edges = []
    rng = np.random.default_rng(42)

    for bucket1, bucket2, count in top_pairs:
        bridges1 = bucket_to_bridges.get(bucket1, [])
        bridges2 = bucket_to_bridges.get(bucket2, [])
        if not bridges1 or not bridges2:
            continue

        # Sample bridge points to avoid too many edges
        sample1 = rng.choice(bridges1, min(max_edges_per_pair, len(bridges1)), replace=False)
        sample2 = rng.choice(bridges2, min(max_edges_per_pair, len(bridges2)), replace=False)

        # Connect sampled bridge points
        for idx1 in sample1:
            for idx2 in sample2:
                edge = {'source': idx_to_node[idx1], 'target': idx_to_node[idx2]}
                if count >= heavy_threshold:
                    heavy_edges.append(edge)
                else:
                    medium_edges.append(edge)

    print(f"  Building {len(medium_edges)} medium + {len(heavy_edges)} heavy point-to-point edges")

    medium_x, medium_y = [], []
    heavy_x, heavy_y = [], []

    # Bundle medium edges
    if medium_edges:
        medium_df = pd.DataFrame(medium_edges)
        bundled = hammer_bundle(nodes_df, medium_df)
        for x, y in zip(bundled['x'], bundled['y']):
            if pd.isna(x):
                medium_x.append(None)
                medium_y.append(None)
            else:
                medium_x.append(x)
                medium_y.append(y)

    # Bundle heavy edges
    if heavy_edges:
        heavy_df = pd.DataFrame(heavy_edges)
        bundled = hammer_bundle(nodes_df, heavy_df)
        for x, y in zip(bundled['x'], bundled['y']):
            if pd.isna(x):
                heavy_x.append(None)
                heavy_y.append(None)
            else:
                heavy_x.append(x)
                heavy_y.append(y)

    return medium_x, medium_y, heavy_x, heavy_y


def cluster_points(
    coords_2d: np.ndarray,
    bucket_ids: List[int],
    n_clusters: int = 12
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Cluster the 2D points for visualization."""
    from sklearn.cluster import KMeans

    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords_2d)

    # Group indices by cluster
    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_to_indices[label].append(idx)

    return cluster_labels, cluster_to_indices


def label_clusters_with_ollama(
    texts: List[str],
    cluster_to_indices: Dict[int, List[int]],
    coords_2d: np.ndarray,
    model: str = "gemma2:9b",
    samples_per_cluster: int = 15
) -> Dict[int, str]:
    """Label clusters using Ollama."""
    if requests is None:
        raise ImportError("requests required: pip install requests")

    print(f"Labeling clusters with {model}...")
    labels = {}

    for cluster_id, indices in cluster_to_indices.items():
        # Sample randomly from cluster
        rng = np.random.default_rng(42 + cluster_id)
        sample_indices = rng.choice(indices, min(samples_per_cluster, len(indices)), replace=False)
        samples = [texts[i] for i in sample_indices]

        prompt = f"""Given these Wikipedia article titles randomly sampled from a cluster, provide a SHORT label (2-4 words max) that describes the common theme:

Articles: {', '.join(samples)}

Reply with ONLY the short label, nothing else."""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={'model': model, 'prompt': prompt, 'stream': False},
                timeout=30
            )
            label = response.json()['response'].strip().strip('"').strip("'")
            labels[cluster_id] = label
            print(f"  Cluster {cluster_id}: {label}")
        except Exception as e:
            labels[cluster_id] = f"Cluster {cluster_id}"
            print(f"  Cluster {cluster_id}: (error: {e})")

    return labels


def create_visualization(
    coords_2d: np.ndarray,
    texts: List[str],
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    cluster_to_indices: Dict[int, List[int]],
    medium_edges: Tuple[List, List],
    heavy_edges: Tuple[List, List],
    centroid_similarities: Optional[np.ndarray] = None,
    bucket_sizes: Optional[np.ndarray] = None,
    title: str = "Wikipedia Knowledge Graph",
    bg_color: str = "#2a2a2a"
) -> go.Figure:
    """Create the Plotly visualization."""
    if go is None:
        raise ImportError("plotly required: pip install plotly")

    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
    ]

    traces = []

    # Medium density edges
    traces.append(go.Scattergl(
        x=medium_edges[0], y=medium_edges[1],
        mode='lines',
        line=dict(color='rgba(120,160,200,0.01)', width=0.6),
        hoverinfo='none',
        name='Medium density bridges'
    ))

    # Heavy density edges
    traces.append(go.Scattergl(
        x=heavy_edges[0], y=heavy_edges[1],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.03)', width=1.5),
        hoverinfo='none',
        name='High density bridges'
    ))

    # Single scatter trace colored by centroid similarity
    if centroid_similarities is not None:
        # Compute sizes from bucket density (log scale, range 2-8)
        if bucket_sizes is not None:
            log_sizes = np.log1p(bucket_sizes)
            min_log, max_log = log_sizes.min(), log_sizes.max()
            if max_log > min_log:
                norm_sizes = (log_sizes - min_log) / (max_log - min_log)
            else:
                norm_sizes = np.ones_like(log_sizes) * 0.5
            sizes = norm_sizes * 6 + 2  # Range: 2 to 8
        else:
            sizes = 4

        # Compute alpha from centroid similarity (range 0.3-0.9) - double encoding
        alphas = np.clip(centroid_similarities, 0, 1) * 0.6 + 0.3

        # Get Viridis colors and apply per-point alpha
        import matplotlib.pyplot as plt
        cmap = plt.cm.viridis
        norm_sim = np.clip(centroid_similarities, 0, 1)
        rgba_colors = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{a:.2f})'
                       for c, a in zip(cmap(norm_sim), alphas)]

        traces.append(go.Scattergl(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers',
            marker=dict(
                color=rgba_colors,
                size=sizes,
                line=dict(width=0),
            ),
            name='Points',
            text=texts,
            hovertemplate='%{text}<extra></extra>'
        ))
    else:
        # Fallback to cluster coloring
        for cluster_id in sorted(cluster_to_indices.keys()):
            indices = cluster_to_indices[cluster_id]
            color = colors[cluster_id % len(colors)]
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

            traces.append(go.Scattergl(
                x=coords_2d[indices, 0],
                y=coords_2d[indices, 1],
                mode='markers',
                marker=dict(color=color, size=4, opacity=0.8, line=dict(width=0)),
                name=f"{name} ({len(indices):,})",
                text=[texts[i] for i in indices],
                hovertemplate='%{text}<extra></extra>'
            ))

    # Cluster annotations
    annotations = []
    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        color = colors[cluster_id % len(colors)]
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        cx = coords_2d[indices, 0].mean()
        cy = coords_2d[indices, 1].mean()
        annotations.append(dict(
            x=cx, y=cy,
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=10, color='white', family='Arial'),
            bgcolor='rgba(30,30,30,0.7)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><sup>{len(texts):,} articles | Bright lines = high-density bridges connecting clusters</sup>',
            font=dict(size=20, color='white', family='Arial'),
            x=0.5, xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20,20,20,0.7)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(color='white', size=10),
            title=dict(text='Clusters (article count)', font=dict(color='#aaa', size=9))
        ),
        hovermode='closest',
        dragmode='pan',
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        width=1400, height=1000,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia knowledge graph visualization')
    parser.add_argument('input', help='Path to embeddings parquet file')
    parser.add_argument('--output', '-o', default='wiki_graph.html', help='Output HTML file')
    parser.add_argument('--n-clusters', type=int, default=12, help='Number of clusters')
    parser.add_argument('--label-with-ollama', action='store_true', help='Label clusters using Ollama')
    parser.add_argument('--ollama-model', default='gemma2:9b', help='Ollama model for labeling')
    parser.add_argument('--title', default='Wikipedia Knowledge Graph', help='Visualization title')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load data
    print(f"Loading embeddings from {args.input}...")
    embeddings, texts, categories = load_embeddings(args.input)
    print(f"  Loaded {len(embeddings):,} embeddings, dim={embeddings.shape[1]}")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Run density classifier
    print("Running density classification...")
    classifier = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=args.seed)
    classifier.fit(embeddings)
    print(f"  {classifier.report()}")

    # Analyze bridges
    print("Analyzing bridges...")
    bridge_analysis = classifier.analyze_bridges(embeddings)
    print(f"  Found {len(bridge_analysis.bridge_indices)} bridge points")
    print(f"  {bridge_analysis.n_buckets} buckets")

    # Project to 2D
    coords_2d = project_to_2d(embeddings, seed=args.seed)

    # Cluster for visualization
    cluster_labels, cluster_to_indices = cluster_points(coords_2d, classifier.get_bucket_ids(), args.n_clusters)

    # Label clusters
    if args.label_with_ollama:
        cluster_names = label_clusters_with_ollama(
            texts, cluster_to_indices, coords_2d,
            model=args.ollama_model
        )
    else:
        cluster_names = {i: f"Cluster {i}" for i in range(args.n_clusters)}

    # Compute edge bundles
    print("Computing edge bundles...")
    medium_x, medium_y, heavy_x, heavy_y = compute_edge_bundles(
        coords_2d, bridge_analysis, classifier
    )
    print(f"  Medium edges: {sum(1 for x in medium_x if x is None)} paths")
    print(f"  Heavy edges: {sum(1 for x in heavy_x if x is None)} paths")

    # Get centroid similarities and bucket sizes
    centroid_similarities = np.array(classifier.get_centroid_similarities())
    bucket_sizes = np.array(classifier.get_bucket_sizes())

    # Create visualization
    print("Creating visualization...")
    fig = create_visualization(
        coords_2d, texts, cluster_labels, cluster_names, cluster_to_indices,
        (medium_x, medium_y), (heavy_x, heavy_y),
        centroid_similarities=centroid_similarities,
        bucket_sizes=bucket_sizes,
        title=args.title
    )

    # Save
    fig.write_html(args.output, config={'scrollZoom': True, 'displayModeBar': True})
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
