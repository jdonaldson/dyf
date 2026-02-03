"""DYF tree: recursive k-ary splitting using PCA-based LSH.

Builds a k-ary tree by recursively splitting points using DensityClassifier
(multi-axis PCA on centroids).  Each split records per-point centroid
similarities as margins, enabling boundary persistence analysis and
hierarchical clustering.

Compared to pca_tree (which splits on a single PC1 axis per level), dyf_tree
splits on multiple PCA-derived axes simultaneously via LSH bucketing.  This
captures multi-dimensional topic boundaries at each level.

Public API:
    build_dyf_tree              — construct the tree from embeddings
    extract_boundary_persistence — find boundary points at multiple depths
    boundary_persistence_scores — convenience: depth-weighted score array
    cut_dyf_tree_to_labels      — cut the tree into flat cluster labels
"""

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_dyf_tree(embeddings, point_indices, depth, num_bits, min_leaf_size,
                    seed):
    """Recursively split points using DYF LSH, storing per-point margins.

    Returns nested dict tree with keys:
        children, indices, depth, point_margin_map
    """
    from dyf_rs import DensityClassifier

    if depth == 0 or len(point_indices) < min_leaf_size * 2:
        return {
            'children': [],
            'indices': point_indices,
            'depth': depth,
            'point_margin_map': None,
        }

    subset = embeddings[point_indices]
    dim = subset.shape[1]

    try:
        clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits, seed=seed)
        clf.fit(subset)
        bucket_ids = np.array(clf.get_bucket_ids())
        centroid_sims = np.array(clf.get_centroid_similarities())
    except Exception:
        return {
            'children': [],
            'indices': point_indices,
            'depth': depth,
            'point_margin_map': None,
        }

    # Use centroid_similarity as margin: low = far from center = boundary
    point_margin_map = {}
    for i, gidx in enumerate(point_indices):
        point_margin_map[int(gidx)] = float(centroid_sims[i])

    # Group by bucket
    unique_buckets = sorted(set(bucket_ids.tolist()))

    # If DYF produced only one bucket, can't split further
    if len(unique_buckets) <= 1:
        return {
            'children': [],
            'indices': point_indices,
            'depth': depth,
            'point_margin_map': point_margin_map,
        }

    children = []
    for bid in unique_buckets:
        mask = bucket_ids == bid
        child_indices = point_indices[mask]
        if len(child_indices) < min_leaf_size:
            # Too small to recurse — make a leaf
            children.append({
                'children': [],
                'indices': child_indices,
                'depth': depth - 1,
                'point_margin_map': None,
            })
        else:
            child = _build_dyf_tree(
                embeddings, child_indices, depth - 1, num_bits,
                min_leaf_size, seed)
            children.append(child)

    return {
        'children': children,
        'indices': point_indices,
        'depth': depth,
        'point_margin_map': point_margin_map,
    }


def build_dyf_tree(embeddings, max_depth, num_bits=3, min_leaf_size=4,
                   seed=42):
    """Build a DYF recursive tree over embeddings.

    At each level, fits a DensityClassifier with ``num_bits`` bits, producing
    up to 2^num_bits children per node.  Centroid similarities are stored as
    per-point margins for boundary persistence analysis.

    Args:
        embeddings: (n, d) array of embedding vectors.
        max_depth: Maximum tree depth (number of recursive splits).
        num_bits: LSH bits per level (default 3 = up to 8-way splits).
        min_leaf_size: Stop splitting when a node has fewer than
                       2 * min_leaf_size points.
        seed: Random seed for DensityClassifier.

    Returns:
        Tree dict with keys: children, indices, depth, point_margin_map.
    """
    embeddings = np.asarray(embeddings)
    all_indices = np.arange(len(embeddings))
    return _build_dyf_tree(
        embeddings, all_indices, max_depth, num_bits, min_leaf_size, seed)


# ---------------------------------------------------------------------------
# Boundary persistence detection
# ---------------------------------------------------------------------------

def extract_boundary_persistence(tree, margin_pct=0.10):
    """Identify points that persist as boundary across multiple tree depths.

    Same concept as pca_tree.extract_boundary_persistence but for k-ary DYF
    trees.  At each depth, points with centroid similarity below the
    margin_pct percentile threshold are tagged as boundary.

    Args:
        tree: DYF tree dict from build_dyf_tree().
        margin_pct: Percentile threshold (0-1).

    Returns:
        dict with:
            boundary_depths: dict[int, list[int]]
            boundary_count: np.ndarray shape (n,)
            thresholds: dict[int, float]
    """
    margins_by_depth = defaultdict(list)
    nodes_by_depth = defaultdict(list)

    def _collect(node, current_depth):
        if node['point_margin_map'] is not None:
            margins_by_depth[current_depth].extend(
                node['point_margin_map'].values())
            nodes_by_depth[current_depth].append(node)
        for child in node['children']:
            _collect(child, current_depth + 1)

    _collect(tree, 0)

    thresholds = {}
    for depth, margins in margins_by_depth.items():
        thresholds[depth] = np.percentile(margins, margin_pct * 100)

    boundary_depths = defaultdict(list)

    for depth, nodes in nodes_by_depth.items():
        threshold = thresholds[depth]
        for node in nodes:
            for pt_idx, margin in node['point_margin_map'].items():
                if margin < threshold:
                    boundary_depths[pt_idx].append(depth)

    n = len(tree['indices'])
    boundary_count = np.zeros(n, dtype=int)
    for pt_idx, depths in boundary_depths.items():
        boundary_count[pt_idx] = len(depths)

    return {
        'boundary_depths': dict(boundary_depths),
        'boundary_count': boundary_count,
        'thresholds': thresholds,
    }


def boundary_persistence_scores(tree, margin_pct=0.10, max_depth=None):
    """Compute depth-weighted boundary persistence bridge scores.

    Args:
        tree: DYF tree dict from build_dyf_tree().
        margin_pct: Percentile threshold (0-1) for boundary detection.
        max_depth: Tree depth used for weighting.  If None, inferred
                   from tree['depth'].

    Returns:
        np.ndarray of shape (n,) with non-negative bridge scores.
    """
    if max_depth is None:
        max_depth = tree['depth']

    result = extract_boundary_persistence(tree, margin_pct=margin_pct)
    boundary_depths = result['boundary_depths']

    n = len(tree['indices'])
    scores = np.zeros(n, dtype=np.float64)
    for pt_idx, depths in boundary_depths.items():
        scores[pt_idx] = sum(max_depth - d for d in depths)

    return scores


# ---------------------------------------------------------------------------
# Cut tree to flat labels
# ---------------------------------------------------------------------------

def _collect_leaves(node):
    """Collect all leaf nodes from a DYF tree."""
    if not node['children']:
        return [node]
    leaves = []
    for child in node['children']:
        leaves.extend(_collect_leaves(child))
    return leaves


def cut_dyf_tree_to_labels(tree, n_points, n_clusters, embeddings):
    """Cut DYF tree into flat cluster labels using agglomerative merge.

    Collects leaf nodes, computes their cosine centroids, then merges
    leaves to n_clusters using agglomerative clustering with cosine
    distance and average linkage.

    Args:
        tree: DYF tree dict from build_dyf_tree().
        n_points: Total number of points.
        n_clusters: Desired number of clusters.
        embeddings: (n, d) array used for centroid computation.

    Returns:
        np.ndarray of shape (n_points,) with cluster labels.
    """
    leaves = _collect_leaves(tree)
    n_leaves = len(leaves)

    # Normalize embeddings for cosine centroids
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-10)

    if n_leaves <= n_clusters:
        # Fewer leaves than target — each leaf is its own cluster
        labels = np.zeros(n_points, dtype=int)
        for i, leaf in enumerate(leaves):
            for p in leaf['indices']:
                labels[p] = i
        return labels

    # Compute cosine centroids per leaf
    dim = embeddings.shape[1]
    centroids = np.zeros((n_leaves, dim), dtype=np.float32)
    for i, leaf in enumerate(leaves):
        cent = emb_n[leaf['indices']].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        centroids[i] = cent

    # Agglomerative merge on cosine distance
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric='cosine', linkage='average')
    leaf_labels = agg.fit_predict(centroids)

    # Map points to cluster labels
    labels = np.zeros(n_points, dtype=int)
    for i, leaf in enumerate(leaves):
        for p in leaf['indices']:
            labels[p] = leaf_labels[i]

    return labels
