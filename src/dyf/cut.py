"""Cut a DYF tree into flat cluster labels.

This was a dispatcher over two tree shapes — ``build_pca_tree`` produced
``{'left', 'right', ...}`` and ``build_dyf_tree`` produces ``{'children', ...}``, and it
routed by sniffing for a key. That existed because the mismatch had already bitten once
(``KNOWN_ISSUES`` #2: a bare ``KeyError: 'left'``).

``pca_tree`` was dropped on 2026-09-05 — nothing in the package produced a PCA tree, and
``dyf_tree`` supersedes it — so there is one tree shape and nothing to route. The shape
check is kept, because passing the wrong dict should still say so rather than fail deep
inside the cut with a ``KeyError``.
"""

from __future__ import annotations

from .dyf_tree import _cut_dyf_tree_to_labels


def cut_tree_to_labels(
    tree,
    n_points,
    n_clusters,
    *,
    max_depth=None,
    embeddings=None,
):
    """Cut a DYF tree into flat cluster labels.

    Args:
        tree: Tree dict from ``build_dyf_tree``.
        n_points: Total number of points.
        n_clusters: Desired number of clusters.
        max_depth: Unused. Accepted so callers written against the two-tree dispatcher
            keep working; it was only ever required for PCA trees.
        embeddings: (n, d) array — required, used for leaf centroids.

    Returns:
        np.ndarray of shape (n_points,) with cluster labels.

    Raises:
        ValueError: If the tree shape is unrecognized or ``embeddings`` is missing.
    """
    if "children" not in tree:
        raise ValueError(
            f"Unrecognized tree shape: expected a 'children' key from build_dyf_tree, "
            f"got {sorted(tree.keys())}. "
            "(build_pca_tree and its 'left'/'right' shape were removed in 0.13.)"
        )
    if embeddings is None:
        raise ValueError("cut_tree_to_labels requires embeddings= to compute leaf centroids")
    return _cut_dyf_tree_to_labels(tree, n_points, n_clusters, embeddings)
