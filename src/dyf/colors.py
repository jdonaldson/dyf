"""Spatial cluster color mapping.

Orders clusters by angular position of their embedding centroids so that
semantically similar clusters get adjacent hues on the color wheel.
"""

import colorsys

import numpy as np


def _hue_order_from_embeddings(labels, embeddings):
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


def spatial_rgb_map(labels, embeddings):
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


def spatial_color_map(labels, embeddings):
    """Return dict mapping label -> hex color with spatially coherent hues."""
    ordered = _hue_order_from_embeddings(labels, embeddings)
    n = len(ordered)
    cmap = {}
    for rank, cid in enumerate(ordered):
        hue = rank / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.6)
        cmap[int(cid)] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return cmap


def golden_ratio_color_map(labels):
    """Return dict mapping label -> hex color (label-order, not spatial)."""
    unique = sorted(set(labels))
    hues = [(i * 0.618033988749895) % 1.0 for i in range(len(unique))]
    cmap = {}
    for i, lbl in enumerate(unique):
        r, g, b = colorsys.hls_to_rgb(hues[i], 0.45, 0.6)
        cmap[int(lbl)] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return cmap
