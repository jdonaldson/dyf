"""Unified tree-cutting dispatcher.

Both ``build_pca_tree`` and ``build_dyf_tree`` produce dicts with different
shapes:

- PCA tree:  ``{'left', 'right', 'indices', 'depth', 'point_margin_map'}``
- DYF tree:  ``{'children', 'indices', 'depth', 'point_margin_map', ...}``

``cut_tree_to_labels`` detects the shape and routes to the correct impl,
so callers don't have to remember which cut function pairs with which
builder.
"""

from __future__ import annotations

from .dyf_tree import _cut_dyf_tree_to_labels
from .pca_tree import _cut_pca_tree_to_labels


def cut_tree_to_labels(
    tree,
    n_points,
    n_clusters,
    *,
    max_depth=None,
    embeddings=None,
):
    """Cut a PCA or DYF tree into flat cluster labels.

    Detects tree shape from its keys and routes to the appropriate impl.

    Args:
        tree: Tree dict from ``build_pca_tree`` or ``build_dyf_tree``.
        n_points: Total number of points.
        n_clusters: Desired number of clusters.
        max_depth: Required for PCA trees — the depth used when building.
        embeddings: Required for DYF trees — (n, d) array for leaf centroids.

    Returns:
        np.ndarray of shape (n_points,) with cluster labels.

    Raises:
        ValueError: If the tree shape is unrecognized, or required kwargs
            for the detected shape are missing.
    """
    if 'children' in tree:
        if embeddings is None:
            raise ValueError(
                "DYF tree (has 'children' key) requires embeddings= kwarg"
            )
        return _cut_dyf_tree_to_labels(tree, n_points, n_clusters, embeddings)
    if 'left' in tree:
        if max_depth is None:
            raise ValueError(
                "PCA tree (has 'left'/'right' keys) requires max_depth= kwarg"
            )
        return _cut_pca_tree_to_labels(tree, max_depth, n_points, n_clusters)
    raise ValueError(
        f"Unrecognized tree shape: expected 'children' (DYF) or 'left' (PCA) "
        f"key, got {sorted(tree.keys())}"
    )
