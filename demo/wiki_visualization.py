"""
Wikipedia Knowledge Graph Visualization

Generates an interactive visualization of Wikipedia article embeddings,
showing clusters and density-based bridges between them.

Requirements:
    pip install dyf umap-learn plotly scikit-learn requests datashader

Usage:
    python wiki_visualization.py embeddings.parquet --output wiki_graph.html
    python wiki_visualization.py embeddings.parquet --label-with-ollama
    python wiki_visualization.py embeddings.parquet --color-mode bridges  # Color by bridge connections
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
from dataclasses import dataclass


@dataclass
class FacetResult:
    """Result of faceting a single dense bucket."""
    parent_bucket_id: int
    facet_ids: np.ndarray
    facet_centroid_sims: np.ndarray
    bridge_indices: List[int]
    n_facets: int
    local_to_global_idx: np.ndarray


@dataclass
class HierarchicalDYF:
    """Two-tier DYF structure with global buckets and local facets."""
    global_bucket_ids: np.ndarray
    global_centroid_sims: np.ndarray
    global_bridge_indices: List[int]
    global_bridge_analysis: BridgeAnalysis
    n_global_buckets: int
    facets: Dict[int, FacetResult]
    dense_bucket_ids: List[int]
    dense_threshold: float
    combined_facet_ids: np.ndarray
    is_in_faceted_bucket: np.ndarray

    def summary(self) -> str:
        n_faceted = self.is_in_faceted_bucket.sum()
        total_facets = sum(f.n_facets for f in self.facets.values())
        facet_bridges = sum(len(f.bridge_indices) for f in self.facets.values())
        return (
            f"Global: {self.n_global_buckets} buckets, {len(self.global_bridge_indices)} bridges | "
            f"Facets: {total_facets} in {len(self.facets)} dense buckets, {facet_bridges} bridges"
        )


def build_hierarchical_dyf(
    embeddings: np.ndarray,
    global_num_bits: int = 12,
    facet_num_bits: int = 10,
    dense_percentile: float = 75,
    min_bucket_size: int = 20,
    seed: int = 42
) -> HierarchicalDYF:
    """Build two-tier hierarchical DYF with global buckets and local facets."""
    n_points, dim = embeddings.shape

    # Global tier
    global_clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
    global_clf.fit(embeddings)

    global_bucket_ids = global_clf.get_bucket_ids()
    global_centroid_sims = global_clf.get_centroid_similarities()
    global_bridge_analysis = global_clf.analyze_bridges(embeddings)
    global_bridge_indices = list(global_bridge_analysis.bridge_indices)

    bucket_counts = np.bincount(global_bucket_ids)
    populated = bucket_counts[bucket_counts > 0]
    n_global_buckets = len(populated)

    dense_threshold = np.percentile(populated, dense_percentile)
    dense_bucket_ids = np.where(bucket_counts > max(dense_threshold, min_bucket_size))[0].tolist()

    bucket_to_indices = defaultdict(list)
    for idx, bid in enumerate(global_bucket_ids):
        bucket_to_indices[bid].append(idx)

    # Facet tier
    facets = {}
    combined_facet_ids = np.full(n_points, -1, dtype=np.int32)
    is_in_faceted_bucket = np.zeros(n_points, dtype=bool)
    facet_id_offset = 0

    for bid in dense_bucket_ids:
        indices = np.array(bucket_to_indices[bid])
        if len(indices) < min_bucket_size:
            continue

        bucket_emb = embeddings[indices]
        bits = 6 if len(indices) < 100 else (8 if len(indices) < 500 else facet_num_bits)

        try:
            facet_clf = DensityClassifier(embedding_dim=dim, num_bits=bits, seed=seed)
            facet_clf.fit(bucket_emb)

            local_ids = facet_clf.get_bucket_ids()
            facet_sims = facet_clf.get_centroid_similarities()
            facet_bridge = facet_clf.analyze_bridges(bucket_emb)

            for local_idx, local_fid in enumerate(local_ids):
                global_idx = indices[local_idx]
                combined_facet_ids[global_idx] = facet_id_offset + local_fid
                is_in_faceted_bucket[global_idx] = True

            facets[bid] = FacetResult(
                parent_bucket_id=bid,
                facet_ids=local_ids,
                facet_centroid_sims=facet_sims,
                bridge_indices=list(facet_bridge.bridge_indices),
                n_facets=len(np.unique(local_ids)),
                local_to_global_idx=indices
            )
            facet_id_offset += local_ids.max() + 1
        except Exception:
            pass

    return HierarchicalDYF(
        global_bucket_ids=global_bucket_ids,
        global_centroid_sims=global_centroid_sims,
        global_bridge_indices=global_bridge_indices,
        global_bridge_analysis=global_bridge_analysis,
        n_global_buckets=n_global_buckets,
        facets=facets,
        dense_bucket_ids=dense_bucket_ids,
        dense_threshold=dense_threshold,
        combined_facet_ids=combined_facet_ids,
        is_in_faceted_bucket=is_in_faceted_bucket
    )


@dataclass
class SuperConnectorResult:
    """Result of finding super connectors in an embedding space."""
    indices: np.ndarray  # Indices of super connectors
    global_centrality: np.ndarray  # Global bridge centrality for all points
    local_centrality: np.ndarray  # Local bridge centrality for all points
    quadrant: np.ndarray  # Quadrant label for all points
    global_threshold: float
    local_threshold: float

    def __len__(self):
        return len(self.indices)

    def summary(self) -> str:
        n_super = len(self.indices)
        n_cross = (self.quadrant == 'Cross-Domain').sum()
        n_specialist = (self.quadrant == 'Domain Specialist').sum()
        n_minor = (self.quadrant == 'Minor Bridge').sum()
        return (
            f"Super Connectors: {n_super} | Cross-Domain: {n_cross} | "
            f"Domain Specialists: {n_specialist} | Minor Bridges: {n_minor}"
        )


def find_super_connectors(
    embeddings: np.ndarray,
    global_num_bits: int = 12,
    facet_num_bits: int = 10,
    dense_percentile: float = 75,
    global_threshold_percentile: float = 50,
    local_threshold_percentile: float = 50,
    min_bucket_size: int = 20,
    seed: int = 42
) -> SuperConnectorResult:
    """
    Find super connectors: points with high centrality in both global and local bridge networks.

    Super connectors are ideal RAG anchor points because they:
    - Bridge across major semantic regions (high global centrality)
    - Connect facets within dense clusters (high local centrality)
    - Provide 10x better coverage efficiency than random anchors

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        global_num_bits: LSH bits for global bucketing
        facet_num_bits: LSH bits for facet bucketing
        dense_percentile: Percentile threshold for dense buckets
        global_threshold_percentile: Percentile for "high" global centrality
        local_threshold_percentile: Percentile for "high" local centrality
        min_bucket_size: Minimum bucket size for faceting
        seed: Random seed

    Returns:
        SuperConnectorResult with indices and centrality data
    """
    n_points, dim = embeddings.shape

    # Global DYF and bridge analysis
    global_clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
    global_clf.fit(embeddings)
    global_buckets = global_clf.get_bucket_ids()
    global_bridge = global_clf.analyze_bridges(embeddings)

    # Compute global centrality
    global_centrality = np.zeros(n_points, dtype=np.int32)
    for i in range(len(global_bridge.bridge_indices)):
        point_idx, _, neighbors = global_bridge.get_bridge_connections(i)
        global_centrality[point_idx] = len(neighbors) + 1

    # Group by bucket for faceting
    bucket_to_indices = defaultdict(list)
    for idx, bid in enumerate(global_buckets):
        bucket_to_indices[bid].append(idx)

    bucket_counts = np.bincount(global_buckets)
    dense_threshold = np.percentile(bucket_counts[bucket_counts > 0], dense_percentile)
    dense_bucket_ids = np.where(bucket_counts > max(dense_threshold, min_bucket_size))[0]

    # Compute local centrality within dense buckets
    local_centrality = np.zeros(n_points, dtype=np.int32)

    for bid in dense_bucket_ids:
        indices = np.array(bucket_to_indices[bid])
        if len(indices) < min_bucket_size:
            continue

        bucket_emb = embeddings[indices]
        bits = 6 if len(indices) < 100 else (8 if len(indices) < 500 else facet_num_bits)

        try:
            facet_clf = DensityClassifier(embedding_dim=dim, num_bits=bits, seed=seed)
            facet_clf.fit(bucket_emb)
            facet_bridge = facet_clf.analyze_bridges(bucket_emb)

            for i in range(len(facet_bridge.bridge_indices)):
                local_idx, _, neighbors = facet_bridge.get_bridge_connections(i)
                local_centrality[indices[local_idx]] = len(neighbors) + 1
        except Exception:
            pass

    # Compute thresholds
    global_nonzero = global_centrality[global_centrality > 0]
    local_nonzero = local_centrality[local_centrality > 0]

    global_thresh = np.percentile(global_nonzero, global_threshold_percentile) if len(global_nonzero) > 0 else 1
    local_thresh = np.percentile(local_nonzero, local_threshold_percentile) if len(local_nonzero) > 0 else 1

    # Classify quadrants
    quadrant = np.full(n_points, 'Regular', dtype=object)
    high_global = global_centrality > global_thresh
    high_local = local_centrality > local_thresh
    is_bridge = (global_centrality > 0) | (local_centrality > 0)

    quadrant[is_bridge & ~high_global & ~high_local] = 'Minor Bridge'
    quadrant[high_global & ~high_local] = 'Cross-Domain'
    quadrant[~high_global & high_local] = 'Domain Specialist'
    quadrant[high_global & high_local] = 'Super Connector'

    super_indices = np.where(quadrant == 'Super Connector')[0]

    return SuperConnectorResult(
        indices=super_indices,
        global_centrality=global_centrality,
        local_centrality=local_centrality,
        quadrant=quadrant,
        global_threshold=float(global_thresh),
        local_threshold=float(local_thresh)
    )


@dataclass
class OrthogonalAnchorResult:
    """Result of orthogonal anchor selection."""
    indices: np.ndarray  # Selected anchor indices
    seed_indices: np.ndarray  # Initial seed indices (e.g., super connectors)
    candidate_source: str  # 'bridges', 'all', or 'custom'

    def __len__(self):
        return len(self.indices)


def select_orthogonal_anchors(
    embeddings: np.ndarray,
    k: int,
    seed_indices: Optional[np.ndarray] = None,
    candidate_indices: Optional[np.ndarray] = None,
    use_bridges: bool = True,
    global_num_bits: int = 12,
    seed: int = 42
) -> OrthogonalAnchorResult:
    """
    Select k maximally spread anchors using greedy farthest-point sampling.

    Achieves ~87% of full-bridge recall with ~22% of anchors by eliminating
    redundancy in anchor placement.

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        k: Number of anchors to select
        seed_indices: Initial seed points (default: super connectors)
        candidate_indices: Pool to select from (default: bridges or all points)
        use_bridges: If True and candidate_indices is None, use bridge points
        global_num_bits: LSH bits for bridge detection
        seed: Random seed

    Returns:
        OrthogonalAnchorResult with selected indices
    """
    n_points, dim = embeddings.shape

    # Get seeds (default: super connectors)
    if seed_indices is None:
        sc_result = find_super_connectors(embeddings, global_num_bits=global_num_bits, seed=seed)
        seed_indices = sc_result.indices

    # Get candidates
    if candidate_indices is None:
        if use_bridges:
            clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
            clf.fit(embeddings)
            bridge_analysis = clf.analyze_bridges(embeddings)
            candidate_indices = np.array(bridge_analysis.bridge_indices)
            candidate_source = 'bridges'
        else:
            candidate_indices = np.arange(n_points)
            candidate_source = 'all'
    else:
        candidate_source = 'custom'

    # Initialize with seeds
    selected = list(seed_indices)
    selected_set = set(selected)
    candidates = [c for c in candidate_indices if c not in selected_set]

    # Initialize min distances from seeds
    min_distances = np.full(n_points, np.inf)
    for s in selected:
        dists = 1 - np.dot(embeddings, embeddings[s])
        min_distances = np.minimum(min_distances, dists)

    # Greedy farthest-point selection
    while len(selected) < k and candidates:
        # Find candidate farthest from all selected
        candidate_dists = min_distances[candidates]
        best_local = np.argmax(candidate_dists)
        best_idx = candidates[best_local]

        # Add to selected
        selected.append(best_idx)
        candidates.pop(best_local)

        # Update min distances
        dists = 1 - np.dot(embeddings, embeddings[best_idx])
        min_distances = np.minimum(min_distances, dists)

    return OrthogonalAnchorResult(
        indices=np.array(selected),
        seed_indices=seed_indices,
        candidate_source=candidate_source
    )


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


def compute_bridge_connections(
    bridge_analysis: BridgeAnalysis,
    n_points: int
) -> np.ndarray:
    """Compute number of bucket connections per point.

    Returns array where connection_counts[i] = number of buckets point i connects to.
    Non-bridge points have 0 connections.
    """
    connection_counts = np.zeros(n_points, dtype=np.int32)
    for i in range(len(bridge_analysis.bridge_indices)):
        point_idx, primary_bucket, neighbor_buckets = bridge_analysis.get_bridge_connections(i)
        connection_counts[point_idx] = len(neighbor_buckets) + 1  # primary + neighbors
    return connection_counts


def compute_weighted_edge_bundles(
    coords_2d: np.ndarray,
    bridge_analysis: BridgeAnalysis,
    classifier: DensityClassifier,
    n_top_pairs: int = 300,
    max_edges_per_pair: int = 5
) -> Tuple[List[Tuple[List[float], List[float]]], List[int], List[Tuple[int, int]]]:
    """Compute edge-bundled paths with weights for variable styling.

    Returns (edge_paths, edge_weights, edge_point_pairs) where:
    - edge_paths: list of (x_coords, y_coords) for each bundled edge
    - edge_weights: connection count for each edge
    - edge_point_pairs: (point_idx1, point_idx2) for each edge
    """
    import pandas as pd
    from datashader.bundling import hammer_bundle

    bucket_ids = classifier.get_bucket_ids()
    bridge_indices = set(bridge_analysis.bridge_indices)

    bucket_to_bridges = defaultdict(list)
    for idx in bridge_indices:
        bucket_to_bridges[bucket_ids[idx]].append(idx)

    top_pairs = bridge_analysis.top_connected_pairs(n_top_pairs)

    all_bridge_indices = sorted(bridge_indices)
    idx_to_node = {idx: i for i, idx in enumerate(all_bridge_indices)}
    nodes_data = [{'x': coords_2d[idx, 0], 'y': coords_2d[idx, 1]} for idx in all_bridge_indices]
    nodes_df = pd.DataFrame(nodes_data)

    all_edges = []
    edge_weights = []
    edge_point_pairs = []
    rng = np.random.default_rng(42)

    for bucket1, bucket2, count in top_pairs:
        bridges1 = bucket_to_bridges.get(bucket1, [])
        bridges2 = bucket_to_bridges.get(bucket2, [])
        if not bridges1 or not bridges2:
            continue
        sample1 = rng.choice(bridges1, min(max_edges_per_pair, len(bridges1)), replace=False)
        sample2 = rng.choice(bridges2, min(max_edges_per_pair, len(bridges2)), replace=False)
        for idx1 in sample1:
            for idx2 in sample2:
                all_edges.append({'source': idx_to_node[idx1], 'target': idx_to_node[idx2]})
                edge_weights.append(count)
                edge_point_pairs.append((int(idx1), int(idx2)))

    if not all_edges:
        return [], [], []

    edges_df = pd.DataFrame(all_edges)
    bundled = hammer_bundle(nodes_df, edges_df)

    # Parse bundled paths
    edge_paths = []
    current_path_x = []
    current_path_y = []

    for x, y in zip(bundled['x'], bundled['y']):
        if pd.isna(x):
            if current_path_x:
                edge_paths.append((current_path_x.copy(), current_path_y.copy()))
                current_path_x = []
                current_path_y = []
        else:
            current_path_x.append(float(x))
            current_path_y.append(float(y))

    if current_path_x:
        edge_paths.append((current_path_x, current_path_y))

    return edge_paths, edge_weights[:len(edge_paths)], edge_point_pairs[:len(edge_paths)]


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


def create_bridge_visualization(
    coords_2d: np.ndarray,
    texts: List[str],
    connection_counts: np.ndarray,
    edge_paths: List[Tuple[List[float], List[float]]],
    edge_weights: List[int],
    edge_point_pairs: List[Tuple[int, int]],
    title: str = "Wikipedia Knowledge Graph",
    bg_color: str = "#2a2a2a"
) -> str:
    """Create visualization colored by bridge connections with hover highlighting.

    Returns HTML string with embedded JavaScript for interactivity.
    """
    if go is None:
        raise ImportError("plotly required: pip install plotly")

    import matplotlib.pyplot as plt

    max_conn = connection_counts.max() if connection_counts.max() > 0 else 1
    cmap = plt.cm.plasma

    # Colors by connection count
    colors = []
    for count in connection_counts:
        if count == 0:
            colors.append('rgba(60,60,70,0.4)')
        else:
            norm = count / max_conn
            c = cmap(norm)
            alpha = 0.5 + 0.4 * norm
            colors.append(f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{alpha:.2f})')

    sizes = np.where(connection_counts > 0, 3 + 5 * (connection_counts / max_conn), 2).tolist()

    # Build point -> edge mapping for JS
    point_to_edges = defaultdict(list)
    for edge_idx, (p1, p2) in enumerate(edge_point_pairs):
        point_to_edges[p1].append(edge_idx)
        point_to_edges[p2].append(edge_idx)

    # Normalize weights for styling
    if edge_weights:
        weights = np.array(edge_weights)
        min_w, max_w = weights.min(), weights.max()
        norm_weights = (weights - min_w) / (max_w - min_w) if max_w > min_w else np.ones_like(weights) * 0.5
    else:
        norm_weights = np.array([])

    fig = go.Figure()

    # Background edges with weighted opacity
    n_bins = 5
    if len(norm_weights) > 0:
        weight_bins = np.digitize(norm_weights, np.linspace(0, 1, n_bins + 1)[1:-1])

        for bin_idx in range(n_bins):
            bin_mask = weight_bins == bin_idx
            if not bin_mask.any():
                continue

            opacity = 0.005 + 0.02 * (bin_idx / (n_bins - 1))
            width = 0.5 + 1.0 * (bin_idx / (n_bins - 1))

            bin_x = []
            bin_y = []
            for i, (px, py) in enumerate(edge_paths):
                if bin_mask[i]:
                    bin_x.extend(list(px) + [None])
                    bin_y.extend(list(py) + [None])

            fig.add_trace(go.Scattergl(
                x=bin_x, y=bin_y,
                mode='lines',
                line=dict(color=f'rgba(180,200,255,{opacity})', width=width),
                hoverinfo='skip',
                showlegend=False
            ))

    # Highlight trace (populated by JS on hover)
    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(255,220,100,0.8)', width=2.5),
        hoverinfo='skip',
        showlegend=False,
        name='highlight'
    ))

    # Points (on top for hover)
    fig.add_trace(go.Scattergl(
        x=coords_2d[:, 0].tolist(),
        y=coords_2d[:, 1].tolist(),
        mode='markers',
        marker=dict(color=colors, size=sizes, line=dict(width=0)),
        text=[f"{texts[i]} ({connection_counts[i]} connections)" for i in range(len(texts))],
        customdata=list(range(len(texts))),
        hovertemplate='%{text}<extra></extra>',
        name='Points'
    ))

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><sup>{len(texts):,} points | Colored by bridge connections | Hover to highlight edges</sup>',
            font=dict(size=20, color='white', family='Arial'),
            x=0.5, xanchor='center'
        ),
        showlegend=False,
        dragmode='pan',
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        width=1400, height=1000,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Generate HTML with custom JS for hover highlighting
    html_content = fig.to_html(config={'scrollZoom': True}, include_plotlyjs=True, full_html=True)

    # Prepare edge data for JS
    edge_data_for_js = [{'x': list(px), 'y': list(py)} for px, py in edge_paths]

    custom_js = f'''
<script>
const edgeData = {json.dumps(edge_data_for_js)};
const pointToEdges = {json.dumps({{str(k): v for k, v in point_to_edges.items()}})};

document.addEventListener('DOMContentLoaded', function() {{
    const plotDiv = document.querySelector('.plotly-graph-div');
    if (!plotDiv) return;

    const numTraces = plotDiv.data.length;
    const highlightTraceIdx = numTraces - 2;

    plotDiv.on('plotly_hover', function(data) {{
        if (!data.points || !data.points[0]) return;
        const pt = data.points[0];
        if (pt.curveNumber !== numTraces - 1) return;

        const pointIdx = pt.customdata;
        const edgeIndices = pointToEdges[String(pointIdx)] || [];
        if (edgeIndices.length === 0) return;

        let hx = [], hy = [];
        for (const ei of edgeIndices) {{
            if (ei < edgeData.length) {{
                hx = hx.concat(edgeData[ei].x, [null]);
                hy = hy.concat(edgeData[ei].y, [null]);
            }}
        }}
        Plotly.restyle(plotDiv, {{x: [hx], y: [hy]}}, [highlightTraceIdx]);
    }});

    plotDiv.on('plotly_unhover', function() {{
        Plotly.restyle(plotDiv, {{x: [[null]], y: [[null]]}}, [highlightTraceIdx]);
    }});
}});
</script>
'''
    html_content = html_content.replace('</body>', custom_js + '</body>')
    return html_content


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia knowledge graph visualization')
    parser.add_argument('input', help='Path to embeddings parquet file')
    parser.add_argument('--output', '-o', default='wiki_graph.html', help='Output HTML file')
    parser.add_argument('--n-clusters', type=int, default=12, help='Number of clusters')
    parser.add_argument('--label-with-ollama', action='store_true', help='Label clusters using Ollama')
    parser.add_argument('--ollama-model', default='gemma2:9b', help='Ollama model for labeling')
    parser.add_argument('--title', default='Wikipedia Knowledge Graph', help='Visualization title')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--color-mode', choices=['centroid', 'bridges'], default='centroid',
                        help='Color mode: centroid (similarity) or bridges (connection count)')
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

    if args.color_mode == 'bridges':
        # Bridge connections visualization with hover highlighting
        print("Computing bridge connections...")
        connection_counts = compute_bridge_connections(bridge_analysis, len(embeddings))
        print(f"  {(connection_counts > 0).sum()} bridge points, max {connection_counts.max()} connections")

        print("Computing weighted edge bundles...")
        edge_paths, edge_weights, edge_point_pairs = compute_weighted_edge_bundles(
            coords_2d, bridge_analysis, classifier
        )
        print(f"  {len(edge_paths)} edge paths")

        print("Creating bridge visualization...")
        html_content = create_bridge_visualization(
            coords_2d, texts, connection_counts,
            edge_paths, edge_weights, edge_point_pairs,
            title=args.title
        )

        with open(args.output, 'w') as f:
            f.write(html_content)
    else:
        # Default centroid similarity visualization
        print("Computing edge bundles...")
        medium_x, medium_y, heavy_x, heavy_y = compute_edge_bundles(
            coords_2d, bridge_analysis, classifier
        )
        print(f"  Medium edges: {sum(1 for x in medium_x if x is None)} paths")
        print(f"  Heavy edges: {sum(1 for x in heavy_x if x is None)} paths")

        centroid_similarities = classifier.get_centroid_similarities()
        bucket_sizes = classifier.get_bucket_sizes()

        print("Creating visualization...")
        fig = create_visualization(
            coords_2d, texts, cluster_labels, cluster_names, cluster_to_indices,
            (medium_x, medium_y), (heavy_x, heavy_y),
            centroid_similarities=centroid_similarities,
            bucket_sizes=bucket_sizes,
            title=args.title
        )

        fig.write_html(args.output, config={'scrollZoom': True, 'displayModeBar': True})

    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
