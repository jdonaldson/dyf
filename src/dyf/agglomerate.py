"""Agglomerate DYF tree leaves into semantic buckets.

Walks the tree to find leaf nodes, computes per-leaf embedding centroids,
then uses complete linkage or Louvain community detection to merge leaves
into agglomerated clusters.  After initial assignment, iteratively reassigns
individual points to their nearest bucket centroid to clean up impure leaves.
"""

from __future__ import annotations

import numpy as np


def _collect_leaf_data(idx):
    """Walk the tree and collect per-leaf centroids, point indices, and metadata.

    Returns:
        Tuple of ``(leaves, leaf_centroids, leaf_point_indices, tree)`` or
        ``None`` when the tree has fewer than two leaves.

        *  ``leaves`` – list of leaf node dicts from tree structure.
        *  ``leaf_centroids`` – list of (D,) float32 arrays, one per leaf.
        *  ``leaf_point_indices`` – list of int arrays of item IDs per leaf.
        *  ``tree`` – raw tree node list from ``idx.get_tree_structure()``.
    """
    tree = idx.get_tree_structure()
    leaves = [n for n in tree if n['is_leaf'] and n['batch_index'] >= 0]

    if len(leaves) < 2:
        return None

    dim = idx.embedding_dim
    is_pq = idx.is_pq
    if is_pq:
        idx._load_pq_codebook()

    leaf_centroids = []
    leaf_point_indices = []

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

    return leaves, leaf_centroids, leaf_point_indices, tree


def _build_item_leaf_map(leaves, leaf_point_indices, n_points):
    """Build item → leaf node_id map (before agglomeration)."""
    item_leaf_map = np.full(n_points, -1, dtype=np.int32)
    for leaf, item_ids in zip(leaves, leaf_point_indices):
        valid = item_ids < n_points
        item_leaf_map[item_ids[valid]] = leaf['node_id']
    return item_leaf_map


def _reassign_points(point_labels, embeddings):
    """Iteratively reassign points to nearest bucket centroid.

    Modifies ``point_labels`` in-place and returns the updated array.
    """
    unique_groups = sorted(set(point_labels.tolist()))
    n_buckets = len(unique_groups)
    gid_to_idx = {gid: i for i, gid in enumerate(unique_groups)}

    # Normalize all point embeddings once
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(emb_norms, 1e-10)

    # Iterative reassignment (converges in 2-3 rounds)
    changed = 0
    for _iter in range(5):
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

        all_sims = emb_normed @ bucket_centroids.T
        best_bucket_idx = np.argmax(all_sims, axis=1)
        new_labels = np.array([unique_groups[bi] for bi in best_bucket_idx],
                              dtype=np.int32)

        changed = int((new_labels != point_labels).sum())
        point_labels = new_labels
        if changed == 0:
            break

    print(f"    Point reassignment: {_iter + 1} iterations, "
          f"{changed} changed in last round")
    return point_labels


def _build_output(point_labels, coords):
    """Build lsh_names and lsh_label_data from final point_labels."""
    unique_groups = sorted(set(point_labels.tolist()))
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
    return lsh_names, lsh_label_data


def merge_to_max_k(point_labels, embeddings, max_k=12):
    """Merge communities to ≤max_k using complete linkage on community centroids.

    Returns ``point_labels`` unchanged if already ≤ max_k.
    Otherwise: compute community centroids → L2-normalize → complete linkage →
    fcluster at max_k → remap point labels → reassign points.

    Args:
        point_labels: int32 array (N,) of cluster IDs.
        embeddings: (N, D) embedding matrix (float32).
        max_k: Maximum number of clusters to keep (default 12).

    Returns:
        int32 array (N,) of merged cluster IDs (0-based contiguous).
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    unique_ids = sorted(set(point_labels.tolist()))
    current_k = len(unique_ids)
    if current_k <= max_k:
        return point_labels

    # Compute community centroids
    dim = embeddings.shape[1]
    centroids = np.zeros((current_k, dim), dtype=np.float32)
    id_to_idx = {gid: i for i, gid in enumerate(unique_ids)}
    for gid in unique_ids:
        mask = point_labels == gid
        centroids[id_to_idx[gid]] = embeddings[mask].mean(axis=0)

    # L2-normalize for cosine-distance linkage
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    # Complete linkage → fcluster at max_k
    Z = linkage(centroids_normed, method='complete')
    merge_labels = fcluster(Z, max_k, criterion='maxclust')  # 1-based

    # Remap point labels: old_gid → merge group (0-based)
    remap = {gid: int(merge_labels[id_to_idx[gid]]) - 1
             for gid in unique_ids}
    merged = np.array([remap[int(g)] for g in point_labels], dtype=np.int32)

    # Reassign points to nearest merged centroid
    merged = _reassign_points(merged, embeddings)

    n_merged = len(set(merged.tolist()))
    print(f"    Merged {current_k} → {n_merged} communities (max_k={max_k})")
    return merged


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

    result = _collect_leaf_data(idx)
    if result is None:
        tree = idx.get_tree_structure()
        return None, {}, [], None, tree

    leaves, leaf_centroids, leaf_point_indices, tree = result
    n_points = coords.shape[0]

    item_leaf_map = _build_item_leaf_map(leaves, leaf_point_indices, n_points)

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

    # Point-level reassignment pass
    point_labels = _reassign_points(point_labels, embeddings)

    lsh_names, lsh_label_data = _build_output(point_labels, coords)
    return point_labels, lsh_names, lsh_label_data, item_leaf_map, tree


def louvain_cluster_leaves(idx, coords, embeddings, leaf_k=10,
                           similarity_threshold=0.5, resolution=1.0):
    """Cluster tree leaves via centroid KNN + Louvain community detection.

    Finds natural communities without requiring a target k.  Builds a KNN
    graph over leaf centroids, filters weak edges by cosine similarity
    threshold, then runs Louvain to discover communities.

    Args:
        idx: An open ``LazyIndex`` handle.
        coords: (N, 2-or-3) UMAP coordinates for every item.
        embeddings: (N, D) embedding matrix (float32).
        leaf_k: Number of nearest neighbors per leaf centroid (default 10).
        similarity_threshold: Minimum cosine similarity to keep an edge
            (default 0.5).
        resolution: Louvain resolution parameter; higher = more communities
            (default 1.0).

    Returns:
        Same tuple as ``agglomerate_tree_leaves``:
        ``(point_labels, lsh_names, lsh_label_data, item_leaf_map,
        tree_structure)``.

        Returns ``(None, {}, [], None, tree)`` when the tree has fewer than
        two leaves.
    """
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors

    result = _collect_leaf_data(idx)
    if result is None:
        tree = idx.get_tree_structure()
        return None, {}, [], None, tree

    leaves, leaf_centroids, leaf_point_indices, tree = result
    n_points = coords.shape[0]

    item_leaf_map = _build_item_leaf_map(leaves, leaf_point_indices, n_points)

    centroids = np.vstack(leaf_centroids).astype(np.float32)

    # L2-normalize centroids for cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    # KNN on normalized centroids (cosine via dot product on unit vectors)
    k = min(leaf_k, len(centroids_normed) - 1)
    if k < 1:
        # Degenerate: only 2 leaves, put everything in one bucket
        k = 1
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(centroids_normed)
    distances, indices = nn.kneighbors(centroids_normed)

    # Build graph: edge if cosine similarity > threshold
    G = nx.Graph()
    G.add_nodes_from(range(len(centroids_normed)))
    for i in range(len(centroids_normed)):
        for j_pos in range(1, distances.shape[1]):  # skip self (pos 0)
            j = indices[i, j_pos]
            sim = 1.0 - distances[i, j_pos]  # cosine distance → similarity
            if sim > similarity_threshold:
                G.add_edge(i, j, weight=sim)

    # Louvain community detection
    communities = nx.community.louvain_communities(
        G, weight='weight', resolution=resolution, seed=42)

    # Map communities → leaf labels (0-based)
    leaf_labels = np.full(len(centroids_normed), -1, dtype=np.int32)
    for comm_id, members in enumerate(communities):
        for leaf_idx in members:
            leaf_labels[leaf_idx] = comm_id

    # Handle isolated nodes (no edges above threshold)
    isolated = leaf_labels == -1
    if isolated.any():
        next_id = leaf_labels.max() + 1
        for i in np.where(isolated)[0]:
            leaf_labels[i] = next_id
            next_id += 1

    n_communities = len(set(leaf_labels.tolist()))
    print(f"    Louvain found {n_communities} communities "
          f"from {len(leaves)} leaves (k={k}, res={resolution})")

    # Map points -> leaf -> community (initial assignment)
    point_labels = np.full(n_points, -1, dtype=np.int32)
    for leaf_idx, item_ids in enumerate(leaf_point_indices):
        group_id = int(leaf_labels[leaf_idx])
        valid = item_ids < n_points
        point_labels[item_ids[valid]] = group_id

    # Handle any unassigned points
    unassigned = point_labels == -1
    if unassigned.any():
        max_id = point_labels.max() + 1
        point_labels[unassigned] = max_id

    # Point-level reassignment pass
    point_labels = _reassign_points(point_labels, embeddings)

    lsh_names, lsh_label_data = _build_output(point_labels, coords)
    return point_labels, lsh_names, lsh_label_data, item_leaf_map, tree
