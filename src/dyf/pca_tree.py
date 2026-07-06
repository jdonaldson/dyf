"""PCA tree: hierarchical bisection along principal components.

Builds a binary tree by recursively splitting points at the median of
their first principal component.  Each split records per-point margins
(distance to the split boundary), enabling boundary persistence analysis
and hierarchical clustering.

Public API:
    build_pca_tree              — construct the tree from embeddings
    extract_boundary_persistence — find boundary points at multiple depths
    boundary_persistence_scores — convenience: depth-weighted score array

For flat cluster labels, use the unified ``dyf.cut_tree_to_labels`` dispatcher.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def _build_pca_tree(embeddings, point_indices, depth, min_leaf_size=2):
    """Recursively bisect points along PC1, storing per-point margins.

    Returns nested dict tree with keys:
        left, right, indices, depth, point_margin_map
    """
    if depth == 0 or len(point_indices) < min_leaf_size * 2:
        return {
            "left": None,
            "right": None,
            "indices": point_indices,
            "depth": depth,
            "point_margin_map": None,
        }

    subset = embeddings[point_indices]

    try:
        pca = PCA(n_components=1)
        projections = pca.fit_transform(subset).ravel()
    except Exception as e:
        if len(point_indices) == len(embeddings):
            # Root-level failure degenerates the whole tree to a single leaf —
            # a valid-looking but useless result. Never benign; say so.
            logger.warning(
                "PCA ROOT split failed — returning a single-leaf tree (all %d points in one cluster). Cause: %s",
                len(point_indices),
                e,
            )
        else:
            logger.debug("PCA split failed at depth %d: %s", depth, e)
        return {
            "left": None,
            "right": None,
            "indices": point_indices,
            "depth": depth,
            "point_margin_map": None,
        }

    median = np.median(projections)
    margins = np.abs(projections - median)

    point_margin_map = {}
    for i, gidx in enumerate(point_indices):
        point_margin_map[int(gidx)] = float(margins[i])

    left_mask = projections <= median
    right_mask = ~left_mask

    if left_mask.all() or right_mask.all():
        mid = len(point_indices) // 2
        left_idx = point_indices[:mid]
        right_idx = point_indices[mid:]
    else:
        left_idx = point_indices[left_mask]
        right_idx = point_indices[right_mask]

    return {
        "left": _build_pca_tree(embeddings, left_idx, depth - 1, min_leaf_size),
        "right": _build_pca_tree(embeddings, right_idx, depth - 1, min_leaf_size),
        "indices": point_indices,
        "depth": depth,
        "point_margin_map": point_margin_map,
    }


def build_pca_tree(embeddings, max_depth, min_leaf_size=2):
    """Build a PCA bisection tree over embeddings.

    Args:
        embeddings: (n, d) array of embedding vectors.
        max_depth: Maximum tree depth (number of recursive splits).
        min_leaf_size: Stop splitting when a node has fewer than
                       2 * min_leaf_size points.

    Returns:
        Tree dict with keys: left, right, indices, depth, point_margin_map.
    """
    embeddings = np.asarray(embeddings)
    all_indices = np.arange(len(embeddings))
    return _build_pca_tree(embeddings, all_indices, max_depth, min_leaf_size)


# ---------------------------------------------------------------------------
# Boundary persistence detection
# ---------------------------------------------------------------------------


def extract_boundary_persistence(tree, margin_pct=0.10):
    """Identify points that persist as boundary across multiple tree depths.

    Walks the built tree once, reading existing point_margin_map at each
    internal node.  At each depth, points with margin below the margin_pct
    percentile threshold are tagged as boundary at that depth.

    Points that are boundary at multiple depths have high boundary
    persistence — they straddle concepts at several levels of the hierarchy
    and are semantically polysemous bridge points.

    Args:
        tree: PCA tree dict from build_pca_tree().
        margin_pct: Percentile threshold (0-1).  Points with margin below
                    this percentile at a given depth are boundary.

    Returns:
        dict with:
            boundary_depths: dict[int, list[int]] — point_idx -> list of
                             depths where the point is boundary
            boundary_count: np.ndarray shape (n,) — number of boundary
                            depths per point
            thresholds: dict[int, float] — margin threshold per depth
    """
    margins_by_depth = defaultdict(list)
    nodes_by_depth = defaultdict(list)

    def _collect(node, current_depth):
        if node["point_margin_map"] is not None:
            margins_by_depth[current_depth].extend(node["point_margin_map"].values())
            nodes_by_depth[current_depth].append(node)
        if node["left"] is not None:
            _collect(node["left"], current_depth + 1)
        if node["right"] is not None:
            _collect(node["right"], current_depth + 1)

    _collect(tree, 0)

    thresholds = {}
    for depth, margins in margins_by_depth.items():
        thresholds[depth] = np.percentile(margins, margin_pct * 100)

    boundary_depths = defaultdict(list)

    for depth, nodes in nodes_by_depth.items():
        threshold = thresholds[depth]
        for node in nodes:
            for pt_idx, margin in node["point_margin_map"].items():
                if margin < threshold:
                    boundary_depths[pt_idx].append(depth)

    n = len(tree["indices"])
    boundary_count = np.zeros(n, dtype=int)
    for pt_idx, depths in boundary_depths.items():
        boundary_count[pt_idx] = len(depths)

    return {
        "boundary_depths": dict(boundary_depths),
        "boundary_count": boundary_count,
        "thresholds": thresholds,
    }


def boundary_persistence_scores(tree, margin_pct=0.10, max_depth=None):
    """Compute depth-weighted boundary persistence bridge scores.

    Convenience wrapper that calls extract_boundary_persistence and returns
    a single score array.  Each boundary depth contributes
    (max_depth - depth) to the score, so boundaries at coarse (shallow)
    levels count more.

    Args:
        tree: PCA tree dict from build_pca_tree().
        margin_pct: Percentile threshold (0-1) for boundary detection.
        max_depth: Tree depth used for weighting.  If None, inferred
                   from tree['depth'].

    Returns:
        np.ndarray of shape (n,) with non-negative bridge scores.
    """
    if max_depth is None:
        max_depth = tree["depth"]

    result = extract_boundary_persistence(tree, margin_pct=margin_pct)
    boundary_depths = result["boundary_depths"]

    n = len(tree["indices"])
    scores = np.zeros(n, dtype=np.float64)
    for pt_idx, depths in boundary_depths.items():
        scores[pt_idx] = sum(max_depth - d for d in depths)

    return scores


# ---------------------------------------------------------------------------
# Tree to linkage matrix (for scipy fcluster)
# ---------------------------------------------------------------------------


def _pca_tree_to_Z(tree, max_depth):
    """Convert PCA tree to scipy Z linkage matrix."""
    leaves = []
    internals = []

    def _collect(node, current_depth):
        if node["left"] is None and node["right"] is None:
            leaves.append((node, current_depth))
        else:
            internals.append((node, current_depth))
            if node["left"] is not None:
                _collect(node["left"], current_depth + 1)
            if node["right"] is not None:
                _collect(node["right"], current_depth + 1)

    _collect(tree, 0)
    n_leaves = len(leaves)

    leaf_id_map = {}
    leaf_count = {}
    leaf_points_list = []
    for i, (leaf, _) in enumerate(leaves):
        leaf_id_map[id(leaf)] = i
        leaf_count[id(leaf)] = 1
        leaf_points_list.append(leaf["indices"])

    internals.sort(key=lambda x: -x[1])
    n_merges = len(internals)
    Z = np.zeros((n_merges, 4))
    node_id_map = dict(leaf_id_map)

    for merge_idx, (node, node_depth) in enumerate(internals):
        left_child = node["left"]
        right_child = node["right"]
        left_id = node_id_map[id(left_child)]
        right_id = node_id_map[id(right_child)]
        distance = float(max_depth - node_depth + 1)
        left_n = leaf_count[id(left_child)]
        right_n = leaf_count[id(right_child)]
        Z[merge_idx, 0] = left_id
        Z[merge_idx, 1] = right_id
        Z[merge_idx, 2] = distance
        Z[merge_idx, 3] = left_n + right_n
        new_id = n_leaves + merge_idx
        node_id_map[id(node)] = new_id
        leaf_count[id(node)] = left_n + right_n

    return Z, leaf_points_list


def _cut_pca_tree_to_labels(tree, max_depth, n_points, n_clusters):
    """Cut PCA tree at n_clusters and return point labels array.

    Internal impl — prefer the unified ``dyf.cut_tree_to_labels`` dispatcher.

    Args:
        tree: PCA tree dict from build_pca_tree() (has 'left'/'right' keys).
        max_depth: Depth used when building the tree.
        n_points: Total number of points.
        n_clusters: Desired number of clusters.

    Returns:
        np.ndarray of shape (n_points,) with cluster labels.
    """
    Z, leaf_points_list = _pca_tree_to_Z(tree, max_depth)
    n_leaves = len(leaf_points_list)

    point_to_leaf = np.zeros(n_points, dtype=int)
    for leaf_id, pts in enumerate(leaf_points_list):
        for p in pts:
            point_to_leaf[p] = leaf_id

    if n_leaves <= n_clusters:
        leaf_labels = np.arange(n_leaves)
    else:
        leaf_labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

    point_labels = np.array([leaf_labels[point_to_leaf[i]] for i in range(n_points)])
    return point_labels
