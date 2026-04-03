"""Spatial cluster color mapping.

Orders clusters by angular position of their embedding centroids so that
semantically similar clusters get adjacent hues on the color wheel.
"""

import colorsys

import numpy as np


def _hue_order_from_embeddings(labels: list[int] | np.ndarray, embeddings: np.ndarray) -> list[int]:
    """Order cluster labels by angular position of their embedding centroids.

    Projects cluster centroids onto PCA-2D, computes polar angle from the
    grand centroid, and returns labels sorted by angle.  Clusters that are
    semantically close in embedding space get adjacent positions in the
    ordering, which translates to similar hues when mapped to a color wheel.
    """
    from sklearn.decomposition import PCA

    labels_arr = np.asarray(labels)
    unique = sorted(set(int(l) for l in labels_arr))
    n = len(unique)
    if n <= 1:
        return unique

    # Compute L2-normalized centroid per cluster
    centroids = np.zeros((n, embeddings.shape[1]), dtype=np.float32)
    for i, cid in enumerate(unique):
        mask = labels_arr == cid
        centroids[i] = embeddings[mask].mean(axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = centroids / norms

    # Project to 2D, compute angle from grand centroid
    if centroids.shape[1] > 2:
        proj = PCA(n_components=2).fit_transform(centroids)
    else:
        proj = centroids[:, :2]

    grand = proj.mean(axis=0)
    angles = np.arctan2(proj[:, 1] - grand[1], proj[:, 0] - grand[0])

    # Sort by angle -> clusters close in embedding space get adjacent hues
    order = np.argsort(angles)
    return [unique[i] for i in order]


def spatial_rgb_map(labels: list[int] | np.ndarray, embeddings: np.ndarray) -> dict[int, list[int]]:
    """Return dict mapping label -> [r, g, b] with spatially coherent hues.

    Clusters that are close in embedding space get similar colors.
    Hues are evenly spaced in the sorted angular order so every cluster
    remains visually distinguishable from its neighbors.
    """
    ordered = _hue_order_from_embeddings(labels, embeddings)
    n = len(ordered)
    cmap = {}
    for rank, cid in enumerate(ordered):
        hue = rank / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.6)
        cmap[int(cid)] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap


def spatial_color_map(labels: list[int] | np.ndarray, embeddings: np.ndarray) -> dict[int, str]:
    """Return dict mapping label -> hex color with spatially coherent hues."""
    ordered = _hue_order_from_embeddings(labels, embeddings)
    n = len(ordered)
    cmap = {}
    for rank, cid in enumerate(ordered):
        hue = rank / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.6)
        cmap[int(cid)] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return cmap


def tree_rgb_map(labels: list[int] | np.ndarray, tree_structure: list[dict], item_leaf_map: np.ndarray) -> dict[int, list[int]]:
    """Assign colors by DFS leaf order of the DYF tree.

    Leaves adjacent in DFS share subtree ancestry, so they get similar hues.
    This produces smooth color gradients within tree subtrees and natural
    discontinuities at subtree boundaries.

    Args:
        labels: per-item cluster/bucket labels (array-like of ints)
        tree_structure: list of node dicts from idx.get_tree_structure()
        item_leaf_map: array of length N mapping item index → leaf node_id
    """
    # Build children map
    children_map = {}
    for node in tree_structure:
        pid = node['parent_id']
        if pid is not None:
            children_map.setdefault(pid, []).append(node['node_id'])

    # DFS from root (node 0) — collect leaf node IDs in visit order
    dfs_leaf_order = []
    stack = [0]
    while stack:
        nid = stack.pop()
        kids = children_map.get(nid, [])
        if not kids:
            dfs_leaf_order.append(nid)
        else:
            # reversed so left child is visited first (stack is LIFO)
            stack.extend(reversed(kids))

    leaf_rank = {nid: rank for rank, nid in enumerate(dfs_leaf_order)}

    labels_arr = np.asarray(labels)
    unique = sorted(set(int(l) for l in labels_arr))
    item_leaf_arr = np.asarray(item_leaf_map)

    # Compute mean DFS rank per bucket (weighted by item count per leaf)
    bucket_ranks = {}
    for cid in unique:
        mask = labels_arr == cid
        leaf_ids = item_leaf_arr[mask]
        # Average the DFS rank of all items' leaves in this bucket
        ranks = np.array([leaf_rank.get(int(lid), 0) for lid in leaf_ids],
                         dtype=np.float64)
        bucket_ranks[cid] = ranks.mean() if len(ranks) > 0 else 0.0

    # Sort buckets by their mean DFS rank, assign evenly-spaced hues
    ordered = sorted(unique, key=lambda c: bucket_ranks[c])
    n = len(ordered)
    cmap = {}
    for rank, cid in enumerate(ordered):
        hue = rank / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.6)
        cmap[int(cid)] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap


