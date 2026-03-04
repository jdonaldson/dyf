"""Agglomerate DYF tree leaves into semantic buckets.

Walks the tree to find leaf nodes, computes per-leaf embedding centroids,
then uses complete linkage to merge leaves into n_groups agglomerated
clusters.  After initial assignment, iteratively reassigns individual
points to their nearest bucket centroid to clean up impure leaves.
"""

from __future__ import annotations

import numpy as np


def agglomerate_tree_leaves(idx, coords, embeddings, n_groups=50):
    """Agglomerate DYF tree leaves into ~n_groups using embedding centroids.

    Walks the tree to find leaf nodes, computes per-leaf embedding centroids,
    then uses complete linkage to merge leaves into n_groups agglomerated
    clusters.  After initial assignment, reassigns individual points to their
    nearest bucket centroid to clean up impure leaves and bad merges.

    Args:
        idx: An open ``LazyIndex`` handle (used to read tree structure and
            leaf batches).
        coords: (N, 2-or-3) UMAP coordinates for every item.
        embeddings: (N, D) embedding matrix (float32).
        n_groups: Target number of agglomerated buckets (default 50).

    Returns:
        Tuple of ``(point_labels, lsh_names, lsh_label_data,
        item_leaf_map, tree_structure)`` ready for ``multi_level_data``.

        *  ``point_labels`` – int32 array (N,) of bucket ids (0-based).
        *  ``lsh_names`` – ``{cid: "Bucket <cid>"}`` placeholder names.
        *  ``lsh_label_data`` – list of dicts with centroid x/y/z, size, cid.
        *  ``item_leaf_map`` – int32 array (N,) mapping each item to its
           tree leaf ``node_id`` (before agglomeration).
        *  ``tree_structure`` – raw tree node list from
           ``idx.get_tree_structure()``.

        Returns ``(None, {}, [], None, tree)`` when the tree has fewer than
        two leaves.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    tree = idx.get_tree_structure()
    leaves = [n for n in tree if n['is_leaf'] and n['batch_index'] >= 0]

    if len(leaves) < 2:
        return None, {}, [], None, tree

    dim = idx.embedding_dim
    is_pq = idx.is_pq
    if is_pq:
        idx._load_pq_codebook()

    # Collect per-leaf centroids and point mappings
    leaf_centroids = []
    leaf_point_indices = []  # list of arrays, one per leaf

    for leaf in leaves:
        batch = idx.get_leaf(leaf['batch_index'])
        item_ids = batch.column('item_index').to_numpy()
        emb_col = batch.column('embedding')
        flat = emb_col.values.to_numpy()
        n_rows = len(emb_col)

        if is_pq:
            meta = idx._get_metadata()
            m = int(meta['pq_n_subquantizers'])
            codes = flat.reshape(n_rows, m)
            leaf_emb = idx._pq_reconstruct(codes)
        else:
            leaf_emb = flat.reshape(n_rows, dim).astype(np.float32)

        centroid = leaf_emb.mean(axis=0)
        leaf_centroids.append(centroid)
        leaf_point_indices.append(item_ids)

    # Build item -> leaf node_id map (before agglomeration)
    n_points = coords.shape[0]
    item_leaf_map = np.full(n_points, -1, dtype=np.int32)
    for leaf, item_ids in zip(leaves, leaf_point_indices):
        valid = item_ids < n_points
        item_leaf_map[item_ids[valid]] = leaf['node_id']

    centroids = np.vstack(leaf_centroids).astype(np.float32)

    # L2-normalize centroids for cosine-distance complete linkage
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    # Agglomerate -- complete linkage refuses merges where the most
    # dissimilar pair across two groups exceeds the threshold, which
    # prevents merging semantically distinct subgroups.
    actual_groups = min(n_groups, len(centroids_normed))
    Z = linkage(centroids_normed, method='complete')
    agg_labels = fcluster(Z, actual_groups, criterion='maxclust')  # 1-based

    # Map points -> leaf -> agglomerated group (initial assignment)
    point_labels = np.full(n_points, -1, dtype=np.int32)
    for leaf_idx, item_ids in enumerate(leaf_point_indices):
        group_id = int(agg_labels[leaf_idx]) - 1  # 0-based
        valid = item_ids < n_points
        point_labels[item_ids[valid]] = group_id

    # Handle any unassigned points
    unassigned = point_labels == -1
    if unassigned.any():
        max_id = point_labels.max() + 1
        point_labels[unassigned] = max_id

    # -- Point-level reassignment pass --
    # Compute bucket centroids, then reassign each point to nearest bucket.
    unique_groups = sorted(set(point_labels.tolist()))
    n_buckets = len(unique_groups)
    gid_to_idx = {gid: i for i, gid in enumerate(unique_groups)}

    # Normalize all point embeddings once
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(emb_norms, 1e-10)

    # Iterative reassignment (converges in 2-3 rounds)
    changed = 0
    for _iter in range(5):
        # Compute bucket centroids from current assignments
        bucket_centroids = np.zeros((n_buckets, embeddings.shape[1]),
                                    dtype=np.float32)
        for gid in unique_groups:
            mask = point_labels == gid
            pts = np.where(mask)[0]
            if len(pts) > 0:
                cent = emb_normed[pts].mean(axis=0)
                norm = np.linalg.norm(cent)
                if norm > 1e-10:
                    cent /= norm
                bucket_centroids[gid_to_idx[gid]] = cent

        # Similarity of every point to every bucket centroid: (N, n_buckets)
        all_sims = emb_normed @ bucket_centroids.T

        # Best bucket for each point
        best_bucket_idx = np.argmax(all_sims, axis=1)
        new_labels = np.array([unique_groups[bi] for bi in best_bucket_idx],
                              dtype=np.int32)

        changed = int((new_labels != point_labels).sum())
        point_labels = new_labels
        if changed == 0:
            break
    print(f"    Point reassignment: {_iter + 1} iterations, "
          f"{changed} changed in last round")

    # Rebuild unique_groups after reassignment (some buckets may have emptied)
    unique_groups = sorted(set(point_labels.tolist()))

    # Build names and label_data
    lsh_names = {gid: f"Bucket {gid}" for gid in unique_groups}
    ndim = coords.shape[1]
    lsh_label_data = []
    for gid in unique_groups:
        mask = point_labels == gid
        pts = np.where(mask)[0]
        centroid = coords[pts].mean(axis=0)
        lsh_label_data.append({
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]) if ndim >= 3 else 0.0,
            "text": f"Bucket {gid}",
            "size": int(mask.sum()),
            "cid": int(gid),
            "leaf_cids": [int(gid)],
        })

    return point_labels, lsh_names, lsh_label_data, item_leaf_map, tree
