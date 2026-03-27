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
    refine_dyf_tree             — re-split incoherent leaves with rotated seeds
    cut_dyf_tree_to_labels      — cut the tree into flat cluster labels
    refine_clusters             — safety net on flat labels
    extract_boundary_persistence — find boundary points at multiple depths
    boundary_persistence_scores — convenience: depth-weighted score array
"""

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_dyf_tree(embeddings, point_indices, depth, num_bits, min_leaf_size,
                    seed, fit_method='raw_pca'):
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
            'hyperplanes': None,
            'bucket_id_to_child': None,
            'eigenvalues': None,
        }

    subset = embeddings[point_indices]
    dim = subset.shape[1]

    try:
        # Try with skip_isolation first (faster), fall back without it
        try:
            clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits, seed=seed, skip_isolation=True)
        except TypeError:
            clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits, seed=seed)
        if fit_method == 'itq':
            clf.fit_itq(subset)
        elif fit_method == 'raw_pca' and hasattr(clf, 'fit_raw_pca'):
            clf.fit_raw_pca(subset)
        else:
            clf.fit(subset)
        bucket_ids = clf.get_bucket_ids()
        centroid_sims = clf.get_centroid_similarities()
    except Exception:
        return {
            'children': [],
            'indices': point_indices,
            'depth': depth,
            'point_margin_map': None,
            'hyperplanes': None,
            'bucket_id_to_child': None,
            'eigenvalues': None,
        }

    # Use centroid_similarity as margin: low = far from center = boundary
    point_margin_map = {}
    for i, gidx in enumerate(point_indices):
        point_margin_map[int(gidx)] = float(centroid_sims[i])

    # Capture hyperplanes from the fitted classifier
    try:
        node_hyperplanes = clf.get_hyperplanes()
    except Exception:
        node_hyperplanes = None

    # Capture eigenvalues
    try:
        ev = clf.get_eigenvalues()
        node_eigenvalues = ev if len(ev) > 0 else None
    except Exception:
        node_eigenvalues = None

    # Group by bucket
    unique_buckets = sorted(set(bucket_ids.tolist()))

    # If DYF produced only one bucket, can't split further
    if len(unique_buckets) <= 1:
        return {
            'children': [],
            'indices': point_indices,
            'depth': depth,
            'point_margin_map': point_margin_map,
            'hyperplanes': None,
            'bucket_id_to_child': None,
            'eigenvalues': None,
        }

    children = []
    bucket_id_to_child = {}
    for child_idx, bid in enumerate(unique_buckets):
        mask = bucket_ids == bid
        child_indices = point_indices[mask]
        bucket_id_to_child[bid] = child_idx
        if len(child_indices) < min_leaf_size:
            # Too small to recurse — make a leaf
            children.append({
                'children': [],
                'indices': child_indices,
                'depth': depth - 1,
                'point_margin_map': None,
                'hyperplanes': None,
                'bucket_id_to_child': None,
                'eigenvalues': None,
            })
        else:
            child = _build_dyf_tree(
                embeddings, child_indices, depth - 1, num_bits,
                min_leaf_size, seed, fit_method)
            children.append(child)

    return {
        'children': children,
        'indices': point_indices,
        'depth': depth,
        'point_margin_map': point_margin_map,
        'hyperplanes': node_hyperplanes,
        'bucket_id_to_child': bucket_id_to_child,
        'eigenvalues': node_eigenvalues,
    }


def build_dyf_tree(embeddings, max_depth, num_bits=3, min_leaf_size=4,
                   seed=42, fit_method='raw_pca'):
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
        fit_method: Fitting method — 'raw_pca' (default, PCA on full data),
                    'pca' (PCA on centroid subset), or 'itq'
                    (iterative quantization for tighter partitions).

    Returns:
        Tree dict with keys: children, indices, depth, point_margin_map.
    """
    embeddings = np.asarray(embeddings)
    all_indices = np.arange(len(embeddings))
    return _build_dyf_tree(
        embeddings, all_indices, max_depth, num_bits, min_leaf_size, seed,
        fit_method)


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

    # Agglomerative merge using Ward linkage on L2-normalized centroids.
    # Ward minimizes within-cluster variance, producing balanced merges.
    # On unit vectors, euclidean distance is monotonic with cosine distance
    # (||a-b||^2 = 2(1 - cos(a,b))), so semantics are preserved.
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    leaf_labels = agg.fit_predict(centroids)

    # Map points to cluster labels
    labels = np.zeros(n_points, dtype=int)
    for i, leaf in enumerate(leaves):
        for p in leaf['indices']:
            labels[p] = leaf_labels[i]

    return labels


# ---------------------------------------------------------------------------
# Refinement helpers
# ---------------------------------------------------------------------------

def _leaf_coherence(indices, emb_normed):
    """Mean cosine similarity to centroid for a set of point indices."""
    subset = emb_normed[indices]
    centroid = subset.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid /= norm
    return float((subset @ centroid).mean())


def _try_resplit(indices, embeddings, num_bits, seed):
    """Try to re-split indices with a DensityClassifier at a given seed.

    Returns (child_index_arrays, point_margin_map) or None if split fails.
    """
    from dyf_rs import DensityClassifier

    subset = embeddings[indices]
    dim = subset.shape[1]
    try:
        clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits, seed=seed, skip_isolation=True)
        clf.fit(subset)
        bucket_ids = clf.get_bucket_ids()
        centroid_sims = clf.get_centroid_similarities()
    except Exception:
        return None

    unique_buckets = sorted(set(bucket_ids.tolist()))
    if len(unique_buckets) <= 1:
        return None

    point_margin_map = {}
    for i, gidx in enumerate(indices):
        point_margin_map[int(gidx)] = float(centroid_sims[i])

    child_arrays = []
    for bid in unique_buckets:
        mask = bucket_ids == bid
        child_arrays.append(indices[mask])

    return child_arrays, point_margin_map


def _refine_recursive(node, embeddings, emb_normed, threshold, num_bits,
                      min_leaf_size, max_retries, seed_offset):
    """Walk tree, re-split incoherent leaves. Returns count of refined leaves."""
    if node['children']:
        count = 0
        for child in node['children']:
            count += _refine_recursive(
                child, embeddings, emb_normed, threshold, num_bits,
                min_leaf_size, max_retries, seed_offset)
        return count

    # Leaf node
    indices = node['indices']
    if len(indices) < min_leaf_size * 2:
        return 0

    parent_coh = _leaf_coherence(indices, emb_normed)
    if parent_coh >= threshold:
        return 0

    best_split = None
    best_weighted_coh = parent_coh

    for attempt in range(max_retries):
        seed = seed_offset + attempt
        result = _try_resplit(indices, embeddings, num_bits, seed)
        if result is None:
            continue
        child_arrays, pmm = result

        # Compute weighted-mean child coherence
        total = sum(len(ca) for ca in child_arrays)
        weighted_coh = sum(
            len(ca) * _leaf_coherence(ca, emb_normed)
            for ca in child_arrays
        ) / total

        if weighted_coh > best_weighted_coh:
            best_weighted_coh = weighted_coh
            best_split = (child_arrays, pmm)

    if best_split is None:
        return 0

    child_arrays, pmm = best_split
    node['point_margin_map'] = pmm
    node['children'] = []
    for ca in child_arrays:
        node['children'].append({
            'children': [],
            'indices': ca,
            'depth': node['depth'] - 1,
            'point_margin_map': None,
            'hyperplanes': None,
            'bucket_id_to_child': None,
            'eigenvalues': None,
        })
    return 1


# ---------------------------------------------------------------------------
# Public refinement API
# ---------------------------------------------------------------------------

def refine_dyf_tree(tree, embeddings, min_coherence=None, num_bits=3,
                    min_leaf_size=4, max_retries=3, seed_offset=1000):
    """Re-split incoherent tree leaves with rotated LSH seeds.

    Walks the tree and finds leaves whose mean cosine similarity to centroid
    is below a threshold.  For each, tries re-splitting with different LSH
    seeds and accepts the split if weighted child coherence improves.

    Modifies the tree in-place.

    Args:
        tree: DYF tree dict from build_dyf_tree().
        embeddings: (n, d) array of embedding vectors.
        min_coherence: Coherence threshold. Leaves below this are candidates
            for re-splitting. If None, uses 25th percentile of leaf coherences.
        num_bits: LSH bits for re-splitting (default 3).
        min_leaf_size: Don't try to split leaves smaller than 2x this.
        max_retries: Number of different seeds to try per leaf.
        seed_offset: Base seed for re-splitting (offset from original tree seed).

    Returns:
        dict with stats: n_refined, n_leaves_before, n_leaves_after,
        coherence_before, coherence_after.
    """
    embeddings = np.asarray(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # Collect leaves and compute coherence
    leaves_before = _collect_leaves(tree)
    n_leaves_before = len(leaves_before)
    coherences = []
    for leaf in leaves_before:
        if len(leaf['indices']) >= 2:
            coherences.append(_leaf_coherence(leaf['indices'], emb_normed))
    coherences = np.array(coherences)

    coherence_before = float(coherences.mean()) if len(coherences) > 0 else 0.0

    if min_coherence is None:
        threshold = float(np.percentile(coherences, 25)) if len(coherences) > 0 else 0.0
    else:
        threshold = float(min_coherence)

    n_refined = _refine_recursive(
        tree, embeddings, emb_normed, threshold, num_bits,
        min_leaf_size, max_retries, seed_offset)

    # Recompute stats
    leaves_after = _collect_leaves(tree)
    coherences_after = []
    for leaf in leaves_after:
        if len(leaf['indices']) >= 2:
            coherences_after.append(_leaf_coherence(leaf['indices'], emb_normed))
    coherences_after = np.array(coherences_after)
    coherence_after = float(coherences_after.mean()) if len(coherences_after) > 0 else 0.0

    return {
        'n_refined': n_refined,
        'n_leaves_before': n_leaves_before,
        'n_leaves_after': len(leaves_after),
        'coherence_before': coherence_before,
        'coherence_after': coherence_after,
    }


def refine_clusters(labels, embeddings, min_coherence=None,
                    min_cluster_size=None, num_bits=6, seed_offset=2000):
    """Post-processing safety net: eject periphery from incoherent clusters.

    For each cluster below the coherence threshold, keeps core points (above
    median similarity to centroid) and ejects periphery into a shared pool.
    The pool is re-split with a rotated LSH seed.  Small resulting clusters
    are iteratively merged with their nearest neighbor until they reach
    min_cluster_size, then any remaining runts are absorbed into the nearest
    large cluster.

    Args:
        labels: (n,) cluster label array.
        embeddings: (n, d) array of embedding vectors.
        min_coherence: Coherence threshold. If None, uses 25th percentile.
        min_cluster_size: Minimum viable cluster size.  If None, uses
            n_points / n_clusters / 3 (one-third of the expected median).
        num_bits: LSH bits for re-splitting the ejected pool.
        seed_offset: Seed for the re-split DensityClassifier.

    Returns:
        np.ndarray of shape (n,) with refined cluster labels.
    """
    from dyf_rs import DensityClassifier

    labels = np.asarray(labels).copy()
    embeddings = np.asarray(embeddings)

    if min_cluster_size is None:
        n_original = len(set(labels.tolist()))
        min_cluster_size = max(10, len(labels) // max(n_original, 1) // 3)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # Per-cluster coherence
    unique_labels = sorted(set(labels.tolist()))
    cluster_coherence = {}
    cluster_members = {}
    for cid in unique_labels:
        members = np.where(labels == cid)[0]
        cluster_members[cid] = members
        if len(members) >= 2:
            cluster_coherence[cid] = _leaf_coherence(members, emb_normed)
        else:
            cluster_coherence[cid] = 1.0  # singleton is coherent

    coh_values = np.array(list(cluster_coherence.values()))
    if min_coherence is None:
        threshold = float(np.percentile(coh_values, 25)) if len(coh_values) > 0 else 0.0
    else:
        threshold = float(min_coherence)

    # Identify incoherent clusters and eject periphery
    ejected_indices = []
    n_ejected_clusters = 0
    for cid in unique_labels:
        if cluster_coherence[cid] >= threshold:
            continue
        members = cluster_members[cid]
        if len(members) < 4:
            continue

        # Compute per-point cosine sim to centroid
        subset = emb_normed[members]
        centroid = subset.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid /= norm
        sims = subset @ centroid

        # Keep core (above median), eject periphery
        median_sim = float(np.median(sims))
        periphery_mask = sims < median_sim
        periphery = members[periphery_mask]
        ejected_indices.extend(periphery.tolist())
        # Mark ejected points with -1 temporarily
        labels[periphery] = -1
        n_ejected_clusters += 1

    if not ejected_indices:
        return labels

    ejected_indices = np.array(ejected_indices)
    n_ejected = len(ejected_indices)

    # Compute coherent cluster centroids (for fallback assignment)
    next_label = max(unique_labels) + 1
    coherent_centroids = {}
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        cent = emb_normed[members].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        coherent_centroids[cid] = cent

    # Re-split the ejected pool
    ejected_emb = embeddings[ejected_indices]
    dim = ejected_emb.shape[1]

    try:
        clf = DensityClassifier(embedding_dim=dim, num_bits=num_bits,
                                seed=seed_offset, skip_isolation=True)
        clf.fit(ejected_emb)
        bucket_ids = clf.get_bucket_ids()
    except Exception:
        # Fallback: assign all ejected to nearest coherent centroid
        bucket_ids = np.zeros(len(ejected_indices), dtype=int)

    # Assign each re-split bucket a temporary label
    unique_buckets = sorted(set(bucket_ids.tolist()))
    for bid in unique_buckets:
        mask = bucket_ids == bid
        global_indices = ejected_indices[np.where(mask)[0]]
        labels[global_indices] = next_label
        next_label += 1

    # Identify large vs small clusters
    all_cids = sorted(set(labels.tolist()))
    all_cids = [c for c in all_cids if c != -1]
    large_cids = []
    small_cids = []
    cluster_cents = {}
    for cid in all_cids:
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        cent = emb_normed[members].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        cluster_cents[cid] = cent
        if len(members) >= min_cluster_size:
            large_cids.append(cid)
        else:
            small_cids.append(cid)

    n_merged_together = 0
    n_merged_into_large = 0

    # Phase 1: iteratively merge nearest small cluster pairs until all
    # merged groups reach min_cluster_size or only one small cluster remains
    cluster_sizes = {}
    for cid in small_cids + large_cids:
        cluster_sizes[cid] = int((labels == cid).sum())

    while len(small_cids) >= 2:
        # Find nearest pair among small clusters (cosine similarity)
        best_sim = -1.0
        best_i, best_j = 0, 1
        for i in range(len(small_cids)):
            for j in range(i + 1, len(small_cids)):
                sim = float(cluster_cents[small_cids[i]] @
                            cluster_cents[small_cids[j]])
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        ci, cj = small_cids[best_i], small_cids[best_j]
        # Merge cj into ci
        labels[labels == cj] = ci
        n_merged_together += 1
        cluster_sizes[ci] = cluster_sizes[ci] + cluster_sizes[cj]
        del cluster_sizes[cj]

        # Recompute centroid for merged cluster
        members_ci = np.where(labels == ci)[0]
        cent = emb_normed[members_ci].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        cluster_cents[ci] = cent

        # Update small/large lists
        small_cids.pop(best_j)  # remove cj first (higher index)
        if cluster_sizes[ci] >= min_cluster_size:
            small_cids.remove(ci)
            large_cids.append(ci)

    # Phase 2: merge remaining small clusters into nearest large cluster
    if small_cids and large_cids:
        for cid in small_cids:
            large_cents = np.array([cluster_cents[c] for c in large_cids])
            sims = cluster_cents[cid] @ large_cents.T
            nearest = large_cids[int(np.argmax(sims))]
            labels[labels == cid] = nearest
            n_merged_into_large += 1

    # Compact labels to 0..k-1
    unique_final = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique_final)}
    labels = np.array([remap[l] for l in labels], dtype=int)

    print(f"  refine_clusters: ejected {n_ejected} pts from "
          f"{n_ejected_clusters} clusters, "
          f"{n_merged_together} small merged together, "
          f"{n_merged_into_large} merged into large")

    return labels
