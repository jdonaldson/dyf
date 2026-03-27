"""
PCA Tree kNN for UMAP — Build a kNN graph from the PCA tree with
bridge edges that reconnect points across each split.

At each internal node, boundary points (low margin from the split) get
explicit kNN edges to their nearest neighbors on the other side. This
preserves tree structure while healing the forced binary cuts.

Usage:
    python demo/pca_tree_knn_umap.py demo/wiki_simple_50k.parquet [--sample 8000]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import umap
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster


# ---------------------------------------------------------------------------
# Step 1: Build PCA tree with margins
# ---------------------------------------------------------------------------

def _build_pca_tree_with_margins(embeddings, point_indices, depth,
                                  min_leaf_size=2, coords_2d=None,
                                  n_candidates=5):
    """Recursively bisect points along a principal component, storing margin.

    If coords_2d is provided, tries the top n_candidates PCs and picks the
    one whose median split maximizes the 2D centroid gap between halves.
    This produces semantically meaningful splits (real PCs of embedding space)
    that are also spatially coherent in the UMAP projection.

    If coords_2d is None, falls back to PC1 (original behavior).

    Returns nested dict tree with keys:
        left, right, indices, depth, point_margin_map
    """
    if depth == 0 or len(point_indices) < min_leaf_size * 2:
        return {
            'left': None, 'right': None,
            'indices': point_indices, 'depth': depth,
            'point_margin_map': None,
        }

    subset = embeddings[point_indices]
    n_comps = min(n_candidates, len(point_indices) - 1, subset.shape[1])
    if coords_2d is not None and n_comps > 1:
        n_comps = max(n_comps, 2)

    try:
        pca = PCA(n_components=n_comps)
        all_projections = pca.fit_transform(subset)  # (n_subset, n_comps)
    except Exception:
        return {
            'left': None, 'right': None,
            'indices': point_indices, 'depth': depth,
            'point_margin_map': None,
        }

    if coords_2d is not None and n_comps > 1:
        # Score each PC by 2D centroid gap when split at median
        subset_2d = coords_2d[point_indices]
        best_score = -1.0
        best_pc = 0

        for pc_idx in range(n_comps):
            proj = all_projections[:, pc_idx]
            med = np.median(proj)
            left_mask = proj <= med
            right_mask = ~left_mask

            if left_mask.all() or right_mask.all():
                continue

            left_centroid = subset_2d[left_mask].mean(axis=0)
            right_centroid = subset_2d[right_mask].mean(axis=0)
            gap = float(np.linalg.norm(left_centroid - right_centroid))

            if gap > best_score:
                best_score = gap
                best_pc = pc_idx

        projections = all_projections[:, best_pc]
    else:
        projections = all_projections[:, 0]

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
        'left': _build_pca_tree_with_margins(
            embeddings, left_idx, depth - 1, min_leaf_size, coords_2d, n_candidates),
        'right': _build_pca_tree_with_margins(
            embeddings, right_idx, depth - 1, min_leaf_size, coords_2d, n_candidates),
        'indices': point_indices,
        'depth': depth,
        'point_margin_map': point_margin_map,
    }


# ---------------------------------------------------------------------------
# Step 1b: Extract multi-address (boundary) points from built tree
# ---------------------------------------------------------------------------

def extract_multi_address(tree, margin_pct=0.10):
    """Identify points that are boundary at multiple PCA tree depths.

    Walks the built tree once, reading existing point_margin_map at each
    internal node. At each depth, points with margin below the margin_pct
    percentile threshold are tagged as boundary at that depth.

    Points that are boundary at multiple depths are "multi-address" — they
    straddle concepts at several levels of the hierarchy and are semantically
    polysemous bridge points.

    Args:
        tree: PCA tree dict from _build_pca_tree_with_margins()
        margin_pct: percentile threshold (0-1). Points with margin below
                    this percentile at a given depth are boundary.

    Returns:
        dict with:
            boundary_depths: dict[int, list[int]] — point_idx → list of depths
                             where the point is boundary
            boundary_count: np.ndarray shape (n,) — number of boundary depths
                            per point
            thresholds: dict[int, float] — margin threshold per depth
    """
    # Collect all margins per depth
    margins_by_depth = defaultdict(list)
    nodes_by_depth = defaultdict(list)

    def _collect(node, current_depth):
        if node['point_margin_map'] is not None:
            margins_by_depth[current_depth].extend(node['point_margin_map'].values())
            nodes_by_depth[current_depth].append(node)
        if node['left'] is not None:
            _collect(node['left'], current_depth + 1)
        if node['right'] is not None:
            _collect(node['right'], current_depth + 1)

    _collect(tree, 0)

    # Compute threshold at each depth
    thresholds = {}
    for depth, margins in margins_by_depth.items():
        thresholds[depth] = np.percentile(margins, margin_pct * 100)

    # Tag boundary points per depth
    boundary_depths = defaultdict(list)  # point_idx → [depths]

    for depth, nodes in nodes_by_depth.items():
        threshold = thresholds[depth]
        for node in nodes:
            for pt_idx, margin in node['point_margin_map'].items():
                if margin < threshold:
                    boundary_depths[pt_idx].append(depth)

    # Compute boundary count array
    # Need total number of points from root
    n = len(tree['indices'])
    boundary_count = np.zeros(n, dtype=int)
    for pt_idx, depths in boundary_depths.items():
        boundary_count[pt_idx] = len(depths)

    return {
        'boundary_depths': dict(boundary_depths),
        'boundary_count': boundary_count,
        'thresholds': thresholds,
    }


# ---------------------------------------------------------------------------
# Step 2: Collect leaf ancestry
# ---------------------------------------------------------------------------

def _collect_leaf_ancestry(tree):
    """DFS generator yielding (leaf_node, ancestry) for each leaf.

    ancestry = list of (ancestor_node, sibling_subtree) from leaf up to root.
    sibling_subtree is the subtree on the OTHER side of the split.
    """
    def _dfs(node, ancestry_so_far):
        if node['left'] is None and node['right'] is None:
            # Leaf — yield with accumulated ancestry
            yield (node, list(ancestry_so_far))
        else:
            # Internal node: recurse left, with right as sibling
            ancestry_so_far.append((node, node['right']))
            yield from _dfs(node['left'], ancestry_so_far)
            ancestry_so_far.pop()

            # Recurse right, with left as sibling
            ancestry_so_far.append((node, node['left']))
            yield from _dfs(node['right'], ancestry_so_far)
            ancestry_so_far.pop()

    yield from _dfs(tree, [])


# ---------------------------------------------------------------------------
# Step 3: Get all leaf points from a subtree (cached)
# ---------------------------------------------------------------------------

def _get_all_leaf_points(subtree, cache):
    """Returns flat array of all point indices in a subtree's leaves. Cached."""
    node_id = id(subtree)
    if node_id in cache:
        return cache[node_id]

    if subtree['left'] is None and subtree['right'] is None:
        result = subtree['indices']
    else:
        left_pts = _get_all_leaf_points(subtree['left'], cache)
        right_pts = _get_all_leaf_points(subtree['right'], cache)
        result = np.concatenate([left_pts, right_pts])

    cache[node_id] = result
    return result


# ---------------------------------------------------------------------------
# Step 4: Build PCA tree kNN
# ---------------------------------------------------------------------------

def build_pca_tree_knn(embeddings, max_depth=10, k=50, margin_pct=0.3,
                       min_leaf_size=2):
    """Build kNN graph from PCA tree with boundary-aware cross-split neighbors.

    At each PCA tree split, points near the boundary (low margin) get
    neighbor candidates from the sibling subtree.

    Args:
        embeddings: (n, d) embedding matrix
        max_depth: PCA tree recursion depth
        k: number of neighbors per point (output)
        margin_pct: percentile threshold — points with margin below
                    this percentile at a given depth are "boundary points"
                    and get cross-split candidates
        min_leaf_size: minimum leaf size for PCA tree

    Returns:
        (knn_indices, knn_dists) of shape (n, k+1) with self at position 0
    """
    n = embeddings.shape[0]
    print(f"Building PCA tree (depth={max_depth}, min_leaf={min_leaf_size})...")
    t0 = time.time()

    all_indices = np.arange(n)
    tree = _build_pca_tree_with_margins(embeddings, all_indices, max_depth, min_leaf_size)
    print(f"  Tree built in {time.time() - t0:.1f}s")

    # Precompute subtree point cache
    subtree_cache = {}

    # Collect all margins per depth to compute thresholds
    print("  Computing margin thresholds per depth...")
    margins_by_depth = defaultdict(list)

    def _collect_margins(node, current_depth):
        if node['point_margin_map'] is not None:
            margins_by_depth[current_depth].extend(node['point_margin_map'].values())
        if node['left'] is not None:
            _collect_margins(node['left'], current_depth + 1)
        if node['right'] is not None:
            _collect_margins(node['right'], current_depth + 1)

    _collect_margins(tree, 0)

    # Compute threshold at each depth: margin_pct-th percentile
    thresholds = {}
    for depth, margins in margins_by_depth.items():
        thresholds[depth] = np.percentile(margins, margin_pct * 100)
        print(f"    depth {depth}: {len(margins)} margins, "
              f"threshold={thresholds[depth]:.4f} "
              f"(median={np.median(margins):.4f})")

    # Build point-to-leaf mapping and collect leaf ancestry
    print("  Collecting leaf ancestry...")
    t1 = time.time()

    # For each point, gather candidate neighbors
    point_candidates = defaultdict(set)  # point_idx -> set of candidate indices

    leaf_count = 0
    for leaf, ancestry in _collect_leaf_ancestry(tree):
        leaf_count += 1
        leaf_pts = set(leaf['indices'].tolist())

        # All leaf-mates are candidates for each point
        for pt in leaf_pts:
            point_candidates[pt].update(leaf_pts)

        # Walk up ancestry (from immediate parent to root)
        for ancestor, sibling in reversed(ancestry):
            # ancestor['depth'] = remaining depth; current_depth = max_depth - remaining
            current_depth = max_depth - ancestor['depth']

            threshold = thresholds.get(current_depth, 0.0)

            margin_map = ancestor['point_margin_map']
            if margin_map is None:
                continue

            # Two categories get cross-split candidates:
            # 1. Boundary points (margin < threshold at this split)
            # 2. Starved points (fewer candidates than k — need connectivity)
            eligible_pts = [
                pt for pt in leaf_pts
                if (margin_map.get(pt, float('inf')) < threshold
                    or len(point_candidates[pt]) < k)
                and len(point_candidates[pt]) <= 2 * k
            ]
            if not eligible_pts:
                continue

            sibling_pts = _get_all_leaf_points(sibling, subtree_cache)
            sibling_list = sibling_pts.tolist()

            for pt in eligible_pts:
                point_candidates[pt].update(sibling_list)

    print(f"  Collected ancestry for {leaf_count} leaves in {time.time() - t1:.1f}s")

    # Stats on candidates
    cand_sizes = [len(v) for v in point_candidates.values()]
    print(f"  Candidate stats: min={min(cand_sizes)}, "
          f"median={int(np.median(cand_sizes))}, "
          f"mean={np.mean(cand_sizes):.0f}, "
          f"max={max(cand_sizes)}")

    # Compute distances and select top-k
    print(f"  Computing distances and selecting top-{k} neighbors...")
    t2 = time.time()

    # Normalize embeddings for cosine distance via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb_normed = embeddings / norms

    knn_indices = np.zeros((n, k + 1), dtype=np.int32)
    knn_dists = np.zeros((n, k + 1), dtype=np.float32)

    for pt in range(n):
        candidates = list(point_candidates[pt] - {pt})  # exclude self

        if len(candidates) == 0:
            # Fallback: no candidates found (shouldn't happen with leaf-mates)
            knn_indices[pt, 0] = pt
            knn_indices[pt, 1:] = pt
            knn_dists[pt, :] = 0.0
            continue

        cand_arr = np.array(candidates, dtype=np.int32)

        # Cosine distance = 1 - cosine_similarity
        # cosine_sim = dot(normed_a, normed_b)
        sims = emb_normed[cand_arr] @ emb_normed[pt]
        dists = 1.0 - sims

        # Select top-k closest
        if len(dists) <= k:
            # Not enough candidates — use all, pad with last
            order = np.argsort(dists)
            sel_idx = cand_arr[order]
            sel_dist = dists[order]

            knn_indices[pt, 0] = pt
            knn_dists[pt, 0] = 0.0
            actual_k = len(sel_idx)
            knn_indices[pt, 1:actual_k + 1] = sel_idx
            knn_dists[pt, 1:actual_k + 1] = sel_dist
            # Pad remaining with last neighbor
            if actual_k < k:
                knn_indices[pt, actual_k + 1:] = sel_idx[-1]
                knn_dists[pt, actual_k + 1:] = sel_dist[-1]
        else:
            # Partition to find top-k without full sort
            top_k_idx = np.argpartition(dists, k)[:k]
            top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]

            knn_indices[pt, 0] = pt
            knn_dists[pt, 0] = 0.0
            knn_indices[pt, 1:] = cand_arr[top_k_idx]
            knn_dists[pt, 1:] = dists[top_k_idx]

    # Clamp negative distances (floating point artifacts from cosine)
    np.clip(knn_dists, 0.0, None, out=knn_dists)

    print(f"  kNN computed in {time.time() - t2:.1f}s")

    # Stats on cross-split neighbors
    # Count how many neighbors come from outside the leaf
    cross_count = 0
    total_neighbors = 0
    leaf_point_sets = {}
    for leaf, _ in _collect_leaf_ancestry(tree):
        pts_set = frozenset(leaf['indices'].tolist())
        for pt in leaf['indices']:
            leaf_point_sets[pt] = pts_set

    for pt in range(n):
        my_leaf = leaf_point_sets.get(pt, frozenset())
        for j in range(1, k + 1):
            neighbor = knn_indices[pt, j]
            total_neighbors += 1
            if neighbor not in my_leaf:
                cross_count += 1

    print(f"  Cross-split neighbors: {cross_count}/{total_neighbors} "
          f"({100 * cross_count / max(total_neighbors, 1):.1f}%)")

    return knn_indices, knn_dists, tree


# ---------------------------------------------------------------------------
# Step 4b: Bridged PCA tree kNN — heal splits with targeted cross-edges
# ---------------------------------------------------------------------------

def build_bridged_pca_tree_knn(embeddings, tree, cluster_labels,
                                k=50, bridge_budget=20,
                                bridges_per_point=3):
    """Build kNN graph from pre-built PCA tree with pre-computed cluster labels.

    Strategy:
    1. Every point gets ALL cluster-mates as kNN candidates (dense within-cluster).
    2. At each internal node, boundary points get cross-side nearest neighbors
       as candidates (sparse between-cluster bridges).
    3. Top-k selected from candidates → mostly within-cluster, some bridges.

    Args:
        embeddings: (n, d) embedding matrix (will be L2-normalized internally)
        tree: PCA tree dict (from _build_pca_tree_with_margins)
        cluster_labels: (n,) array of cluster labels per point
        k: neighbors per point in output
        bridge_budget: boundary points per side per split for cross-cluster edges
        bridges_per_point: cross-side neighbors per boundary point

    Returns:
        (knn_indices, knn_dists) shape (n, k+1) with self at position 0
    """
    n = embeddings.shape[0]
    n_clusters = len(set(cluster_labels))
    print(f"Building bridged PCA tree kNN (k={k}, {n_clusters} clusters, "
          f"budget={bridge_budget}, bpp={bridges_per_point})...")

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # Cluster-mates → dense within-cluster candidates
    cluster_members = defaultdict(list)
    for pt in range(n):
        cluster_members[int(cluster_labels[pt])].append(pt)

    point_candidates = defaultdict(set)
    for members in cluster_members.values():
        member_set = set(members)
        for pt in members:
            point_candidates[pt] = set(member_set)  # copy

    cluster_sizes = [len(m) for m in cluster_members.values()]
    print(f"  Cluster sizes: {min(cluster_sizes)}-{max(cluster_sizes)}, "
          f"median={int(np.median(cluster_sizes))}")

    # Walk internal nodes, add sparse cross-cluster bridges
    subtree_cache = {}
    bridge_count_by_depth = defaultdict(int)
    total_bridges = 0

    def _add_bridges(node, current_depth):
        nonlocal total_bridges

        if node['left'] is None or node['right'] is None:
            return

        margin_map = node['point_margin_map']
        if margin_map is None:
            _add_bridges(node['left'], current_depth + 1)
            _add_bridges(node['right'], current_depth + 1)
            return

        left_pts = _get_all_leaf_points(node['left'], subtree_cache)
        right_pts = _get_all_leaf_points(node['right'], subtree_cache)

        # Fixed small budget per split — just enough for connectivity
        budget = min(bridge_budget, len(left_pts), len(right_pts))

        def _boundary_pts(pts, b):
            margins = [(margin_map.get(int(p), float('inf')), int(p))
                       for p in pts]
            margins.sort()
            return np.array([p for _, p in margins[:b]])

        left_boundary = _boundary_pts(left_pts, budget)
        right_boundary = _boundary_pts(right_pts, budget)

        if len(left_boundary) == 0 or len(right_boundary) == 0:
            _add_bridges(node['left'], current_depth + 1)
            _add_bridges(node['right'], current_depth + 1)
            return

        bridges_before = total_bridges

        # Left boundary searches against ALL right points for best matches
        sims = emb_normed[left_boundary] @ emb_normed[right_pts].T
        k_br = min(bridges_per_point, len(right_pts))
        if k_br < len(right_pts):
            top_right = np.argpartition(-sims, k_br, axis=1)[:, :k_br]
        else:
            top_right = np.tile(np.arange(len(right_pts)),
                                (len(left_boundary), 1))

        for i in range(len(left_boundary)):
            lp = int(left_boundary[i])
            for j in top_right[i]:
                rp = int(right_pts[j])
                point_candidates[lp].add(rp)
                point_candidates[rp].add(lp)
                total_bridges += 1

        # Right boundary searches against ALL left points
        sims = emb_normed[right_boundary] @ emb_normed[left_pts].T
        k_br = min(bridges_per_point, len(left_pts))
        if k_br < len(left_pts):
            top_left = np.argpartition(-sims, k_br, axis=1)[:, :k_br]
        else:
            top_left = np.tile(np.arange(len(left_pts)),
                               (len(right_boundary), 1))

        for i in range(len(right_boundary)):
            rp = int(right_boundary[i])
            for j in top_left[i]:
                lp = int(left_pts[j])
                point_candidates[rp].add(lp)
                point_candidates[lp].add(rp)

        bridge_count_by_depth[current_depth] += total_bridges - bridges_before

        _add_bridges(node['left'], current_depth + 1)
        _add_bridges(node['right'], current_depth + 1)

    print("  Adding sparse cross-cluster bridges...")
    t1 = time.time()
    _add_bridges(tree, 0)
    print(f"  {total_bridges} bridge edges added in {time.time() - t1:.1f}s")
    for d in sorted(bridge_count_by_depth):
        print(f"    depth {d}: {bridge_count_by_depth[d]} bridges")

    # Stats on candidate set sizes
    cand_sizes = [len(v) for v in point_candidates.values()]
    print(f"  Candidate stats: min={min(cand_sizes)}, "
          f"median={int(np.median(cand_sizes))}, "
          f"mean={np.mean(cand_sizes):.0f}, "
          f"max={max(cand_sizes)}")

    # Compute distances and select top-k
    print(f"  Computing distances and selecting top-{k} neighbors...")
    t2 = time.time()

    knn_indices = np.zeros((n, k + 1), dtype=np.int32)
    knn_dists = np.zeros((n, k + 1), dtype=np.float32)

    for pt in range(n):
        candidates = list(point_candidates[pt] - {pt})

        if len(candidates) == 0:
            knn_indices[pt, 0] = pt
            knn_indices[pt, 1:] = pt
            knn_dists[pt, :] = 0.0
            continue

        cand_arr = np.array(candidates, dtype=np.int32)
        sims = emb_normed[cand_arr] @ emb_normed[pt]
        dists = 1.0 - sims

        if len(dists) <= k:
            order = np.argsort(dists)
            sel_idx = cand_arr[order]
            sel_dist = dists[order]
            knn_indices[pt, 0] = pt
            knn_dists[pt, 0] = 0.0
            actual_k = len(sel_idx)
            knn_indices[pt, 1:actual_k + 1] = sel_idx
            knn_dists[pt, 1:actual_k + 1] = sel_dist
            if actual_k < k:
                knn_indices[pt, actual_k + 1:] = sel_idx[-1]
                knn_dists[pt, actual_k + 1:] = sel_dist[-1]
        else:
            top_k_idx = np.argpartition(dists, k)[:k]
            top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
            knn_indices[pt, 0] = pt
            knn_dists[pt, 0] = 0.0
            knn_indices[pt, 1:] = cand_arr[top_k_idx]
            knn_dists[pt, 1:] = dists[top_k_idx]

    np.clip(knn_dists, 0.0, None, out=knn_dists)
    print(f"  kNN computed in {time.time() - t2:.1f}s")

    # Check connectivity and fix disconnected components
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components

    adj = lil_matrix((n, n), dtype=np.float32)
    for pt in range(n):
        for j in range(1, k + 1):
            nb = knn_indices[pt, j]
            if nb != pt:
                adj[pt, nb] = 1
                adj[nb, pt] = 1

    n_components, comp_labels = connected_components(adj.tocsr(), directed=False)
    print(f"  Connected components: {n_components}")

    if n_components > 1:
        # Connect components by finding closest cross-component pairs
        # Use component representatives (random sample per component)
        comp_members = defaultdict(list)
        for pt, cl in enumerate(comp_labels):
            comp_members[cl].append(pt)

        comp_sizes = {c: len(m) for c, m in comp_members.items()}
        print(f"  Component sizes: {sorted(comp_sizes.values(), reverse=True)[:10]}")

        # Greedily connect smallest components to largest
        largest_comp = max(comp_sizes, key=comp_sizes.get)
        edges_added = 0

        for comp_id in sorted(comp_sizes, key=comp_sizes.get):
            if comp_id == largest_comp:
                continue

            # Sample from this component and the largest
            comp_pts = np.array(comp_members[comp_id])
            main_pts = np.array(comp_members[largest_comp])

            # Sample at most 50 from each for efficiency
            rng = np.random.default_rng(42)
            if len(comp_pts) > 50:
                comp_sample = comp_pts[rng.choice(len(comp_pts), 50, replace=False)]
            else:
                comp_sample = comp_pts
            if len(main_pts) > 200:
                main_sample = main_pts[rng.choice(len(main_pts), 200, replace=False)]
            else:
                main_sample = main_pts

            # Find closest pair
            sims = emb_normed[comp_sample] @ emb_normed[main_sample].T
            best_i, best_j = np.unravel_index(np.argmax(sims), sims.shape)
            cp = int(comp_sample[best_i])
            mp = int(main_sample[best_j])
            dist = float(1.0 - sims[best_i, best_j])

            # Replace the farthest neighbor of cp with mp
            farthest = np.argmax(knn_dists[cp, 1:]) + 1
            knn_indices[cp, farthest] = mp
            knn_dists[cp, farthest] = max(dist, 0.0)
            # And vice versa
            farthest = np.argmax(knn_dists[mp, 1:]) + 1
            knn_indices[mp, farthest] = cp
            knn_dists[mp, farthest] = max(dist, 0.0)
            edges_added += 1

        print(f"  Added {edges_added} edges to connect components")

        # Re-sort each modified point's neighbors by distance
        for pt in range(n):
            order = np.argsort(knn_dists[pt, 1:]) + 1
            knn_indices[pt, 1:] = knn_indices[pt, order]
            knn_dists[pt, 1:] = knn_dists[pt, order]

    starved = sum(1 for pt in range(n) if len(point_candidates[pt]) <= k)
    print(f"  Starved points (<= k candidates): {starved}/{n} ({100*starved/n:.1f}%)")

    return knn_indices, knn_dists


def nn_descent_refine(knn_indices, knn_dists, embeddings, n_iters=2):
    """Refine kNN graph via nn-descent: check neighbors-of-neighbors.

    For each point, gathers all neighbors-of-neighbors as new candidates.
    If any are closer than the current worst neighbor, swap them in.
    Bridges act as seeds that propagate cross-cluster information.
    """
    n, kp1 = knn_indices.shape
    k = kp1 - 1

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    for iteration in range(n_iters):
        improved = 0
        total_new = 0

        for pt in range(n):
            # Current neighbors (including self at 0)
            current_set = set(int(x) for x in knn_indices[pt])

            # Gather neighbors-of-neighbors: for each neighbor, get THEIR k neighbors
            nbs = knn_indices[pt, 1:]  # shape (k,)
            nob = knn_indices[nbs, 1:]  # shape (k, k) — neighbors of each neighbor
            candidates = np.unique(nob.ravel())

            # Remove self and current neighbors
            candidates = np.array([c for c in candidates if int(c) not in current_set],
                                  dtype=np.int32)

            if len(candidates) == 0:
                continue

            # Compute cosine distances to candidates
            sims = emb_normed[candidates] @ emb_normed[pt]
            new_dists = (1.0 - sims).astype(np.float32)

            # Check if any beat current worst neighbor
            worst = knn_dists[pt, -1]
            better_mask = new_dists < worst
            if not better_mask.any():
                continue

            # Merge new better candidates with existing neighbors, keep top-k
            better_idx = candidates[better_mask]
            better_dists = new_dists[better_mask]
            total_new += len(better_idx)

            all_idx = np.concatenate([knn_indices[pt, 1:], better_idx])
            all_dists = np.concatenate([knn_dists[pt, 1:], better_dists])
            order = np.argsort(all_dists)[:k]
            knn_indices[pt, 1:] = all_idx[order]
            knn_dists[pt, 1:] = all_dists[order]
            improved += 1

        print(f"  nn-descent iter {iteration + 1}: "
              f"{improved}/{n} points improved, {total_new} new neighbors added")

    return knn_indices, knn_dists


# ---------------------------------------------------------------------------
# Step 5: PCA tree to scipy linkage (for cutting into clusters)
# ---------------------------------------------------------------------------

def _pca_tree_to_Z(tree, max_depth):
    """Convert PCA tree to scipy Z linkage matrix (same as rog_preprocess)."""
    leaves = []
    internals = []

    def _collect(node, current_depth):
        if node['left'] is None and node['right'] is None:
            leaves.append((node, current_depth))
        else:
            internals.append((node, current_depth))
            if node['left'] is not None:
                _collect(node['left'], current_depth + 1)
            if node['right'] is not None:
                _collect(node['right'], current_depth + 1)

    _collect(tree, 0)
    n_leaves = len(leaves)

    leaf_id_map = {}
    leaf_count = {}
    leaf_points_list = []
    for i, (leaf, _) in enumerate(leaves):
        leaf_id_map[id(leaf)] = i
        leaf_count[id(leaf)] = 1
        leaf_points_list.append(leaf['indices'])

    internals.sort(key=lambda x: -x[1])
    n_merges = len(internals)
    Z = np.zeros((n_merges, 4))
    node_id_map = dict(leaf_id_map)

    for merge_idx, (node, node_depth) in enumerate(internals):
        left_child = node['left']
        right_child = node['right']
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


# ---------------------------------------------------------------------------
# Visualization helpers (matching compare_2d_birch_vs_pca.py style)
# ---------------------------------------------------------------------------

def build_colors_golden_ratio(labels_arr, micro_arr):
    """Golden-ratio hierarchical coloring matching rog_panel."""
    import colorsys
    n_top = len(set(labels_arr))
    base_hues = [(i * 0.618033988749895) % 1.0 for i in range(n_top)]

    colors = []
    for i in range(len(labels_arr)):
        top_cluster = int(labels_arr[i])
        micro_cluster = int(micro_arr[i])
        base_hue = base_hues[top_cluster % len(base_hues)]
        hue_variation = ((micro_cluster % 10) - 5) * 0.016
        hue = (base_hue + hue_variation) % 1.0
        sat = 0.4 + (micro_cluster % 7) * 0.09
        light = 0.35 + (micro_cluster % 5) * 0.12
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def build_figure(coords, titles_list, labels_arr, colors, cluster_names,
                 centroids, title_str):
    """Build Plotly figure matching the existing comparison style."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(size=4, color=colors, opacity=0.7),
        text=[f"{titles_list[i]}<br>Cluster: {cluster_names[labels_arr[i]]}"
              for i in range(len(titles_list))],
        hoverinfo="text",
        showlegend=False,
    ))
    for cid, name in cluster_names.items():
        mask = labels_arr == cid
        if not mask.any():
            continue
        cx, cy = centroids[cid]
        fig.add_annotation(
            x=cx, y=cy, text=name[:25], showarrow=False,
            font=dict(size=9, color="white", family="Arial Black"),
            bgcolor="rgba(40,40,40,0.85)",
            bordercolor="rgba(100,100,100,0.6)",
            borderwidth=1, borderpad=3,
        )
    fig.update_layout(
        title=title_str,
        plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title="",
                   scaleanchor="x", scaleratio=1),
        dragmode="pan",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def cut_tree_to_labels(tree, max_depth, n_points, n_clusters):
    """Cut PCA tree at n_clusters and return point_labels array."""
    Z, leaf_points_list = _pca_tree_to_Z(tree, max_depth)
    n_leaves = len(leaf_points_list)

    point_to_leaf = np.zeros(n_points, dtype=int)
    for leaf_id, pts in enumerate(leaf_points_list):
        for p in pts:
            point_to_leaf[p] = leaf_id

    if n_leaves <= n_clusters:
        leaf_labels = np.arange(n_leaves)
        print(f"  Only {n_leaves} leaves (< {n_clusters}), using leaf IDs")
    else:
        leaf_labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1

    point_labels = np.array([leaf_labels[point_to_leaf[i]] for i in range(n_points)])
    return point_labels


def cut_tree_adaptive(tree, embeddings, sim_threshold=0.45,
                      min_cluster_size=10, max_sample=200):
    """Cut PCA tree adaptively: stop splitting when a node is coherent.

    Walks top-down.  If a node's mean pairwise cosine similarity exceeds
    sim_threshold, it becomes a cluster (dense region).  Otherwise recurse.
    Leaf nodes that never reach threshold are kept as-is.

    Returns (point_labels, cluster_info) where cluster_info is a list of
    dicts with 'indices', 'sim', 'depth' per cluster.
    """
    # Normalize embeddings for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-10)

    clusters = []
    rng = np.random.default_rng(42)

    def _node_sim(idx):
        """Mean pairwise cosine similarity for a set of point indices."""
        sub = emb_n[idx]
        nc = len(idx)
        if nc < 2:
            return 1.0
        if nc > max_sample:
            sample = rng.choice(nc, max_sample, replace=False)
            sub = sub[sample]
            nc = max_sample
        sim_matrix = sub @ sub.T
        mask_tri = np.triu(np.ones((nc, nc), dtype=bool), k=1)
        return float(sim_matrix[mask_tri].mean())

    def _walk(node, current_depth):
        idx = node['indices']

        # Leaf or too small to split further
        if node['left'] is None or len(idx) < min_cluster_size:
            sim = _node_sim(idx)
            clusters.append({'indices': idx, 'sim': sim, 'depth': current_depth})
            return

        sim = _node_sim(idx)
        if sim >= sim_threshold:
            # Dense region — stop here
            clusters.append({'indices': idx, 'sim': sim, 'depth': current_depth})
        else:
            # Not coherent yet — keep splitting
            _walk(node['left'], current_depth + 1)
            _walk(node['right'], current_depth + 1)

    _walk(tree, 0)

    # Assign labels
    n = len(embeddings)
    point_labels = np.full(n, -1, dtype=int)
    for cid, cl in enumerate(clusters):
        for p in cl['indices']:
            point_labels[p] = cid

    # Stats
    sims = [c['sim'] for c in clusters]
    sizes = [len(c['indices']) for c in clusters]
    depths = [c['depth'] for c in clusters]
    print(f"  Adaptive cut (threshold={sim_threshold}): {len(clusters)} clusters")
    print(f"    Sizes: {min(sizes)}-{max(sizes)}, median={int(np.median(sizes))}")
    print(f"    Depths: {min(depths)}-{max(depths)}, median={int(np.median(depths))}")
    print(f"    Intra-sim: {min(sims):.3f}-{max(sims):.3f}, "
          f"mean={np.mean(sims):.3f}, median={np.median(sims):.3f}")

    # Merge tiny clusters into nearest large cluster
    min_viable = min_cluster_size
    large_clusters = [c for c in clusters if len(c['indices']) >= min_viable]
    small_clusters = [c for c in clusters if len(c['indices']) < min_viable]

    if large_clusters and small_clusters:
        # Compute normalized centroids of large clusters
        norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_n2 = embeddings / np.maximum(norms_e, 1e-10)

        large_centroids = []
        for lc in large_clusters:
            centroid = emb_n2[lc['indices']].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-10)
            large_centroids.append(centroid)
        large_centroids = np.array(large_centroids)

        # Assign each small-cluster point to nearest large cluster
        for sc in small_clusters:
            sc_centroid = emb_n2[sc['indices']].mean(axis=0)
            sc_centroid /= max(np.linalg.norm(sc_centroid), 1e-10)
            sims = large_centroids @ sc_centroid
            nearest = int(np.argmax(sims))
            large_clusters[nearest]['indices'] = np.concatenate([
                large_clusters[nearest]['indices'], sc['indices']])

        # Reassign labels
        point_labels = np.full(n, -1, dtype=int)
        for cid, cl in enumerate(large_clusters):
            for p in cl['indices']:
                point_labels[p] = cid
        clusters = large_clusters

        merged_sizes = [len(c['indices']) for c in clusters]
        print(f"  Merged {len(small_clusters)} small clusters → "
              f"{len(large_clusters)} final clusters")
        print(f"    Final sizes: {min(merged_sizes)}-{max(merged_sizes)}, "
              f"median={int(np.median(merged_sizes))}")

    return point_labels, clusters


def labels_to_names_centroids(coords, titles_list, labels):
    """Compute cluster names (nearest-to-centroid title) and centroids."""
    names = {}
    centroids = {}
    for cid in set(labels):
        mask = labels == cid
        pts = np.where(mask)[0]
        centroid = coords[pts].mean(axis=0)
        centroids[cid] = centroid
        dists = np.linalg.norm(coords[pts] - centroid, axis=1)
        names[cid] = titles_list[pts[np.argmin(dists)]]
    return names, centroids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_umap(embeddings, n_neighbors=15, precomputed_knn=None):
    """Run UMAP and return normalized 2D coords."""
    kwargs = dict(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_jobs=-1,
        verbose=True,
    )
    if precomputed_knn is not None:
        kwargs['precomputed_knn'] = precomputed_knn

    reducer = umap.UMAP(**kwargs)
    coords = np.asarray(reducer.fit_transform(embeddings))

    # Fix NaN coords
    nan_mask = np.isnan(coords).any(axis=1)
    if nan_mask.any():
        print(f"  Replacing {nan_mask.sum()} NaN coords")
        from sklearn.neighbors import NearestNeighbors
        valid = ~nan_mask
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[valid])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        coords[nan_mask] = coords[valid][idx.ravel()]

    # Normalize: median-center + MAD scaling
    median = np.nanmedian(coords, axis=0)
    mad = np.nanmedian(np.abs(coords - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords = (coords - median) / scale
    return coords


def main():
    import subprocess
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from rog_preprocess import suggest_n_neighbors

    parser = argparse.ArgumentParser(
        description="Compare standard UMAP vs adaptive PCA tree kNN UMAP")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--k", type=int, default=50,
                        help="k for PCA tree kNN")
    parser.add_argument("--sim-threshold", type=float, default=0.45,
                        help="Intra-sim threshold for adaptive cutting")
    parser.add_argument("--bridge-budget", type=int, default=20,
                        help="Boundary points per side per split")
    parser.add_argument("--bridges-per-point", type=int, default=3,
                        help="Cross-side neighbors per boundary point")
    parser.add_argument("--nn-descent-iters", type=int, default=2,
                        help="nn-descent refinement iterations (0 to disable)")
    args = parser.parse_args()

    outdir = Path(args.parquet_path).parent

    # ── Load & dedup ─────────────────────────────────────────────────────
    print(f"Loading {args.parquet_path}...")
    df = pl.read_parquet(args.parquet_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, seed=42)

    titles_all = df["title"].to_list()
    embeddings_all = np.array(df["embedding"].to_list())

    from dyf_rs import DensityClassifier as RustClassifier
    from dyf.chunks import deduplicate_chunks

    clf = RustClassifier(embedding_dim=embeddings_all.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings_all.astype(np.float32))
    bucket_ids = clf.get_bucket_ids()
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles_all))

    titles = [t for t, keep in zip(titles_all, dedup_mask) if keep]
    embeddings = embeddings_all[dedup_mask]
    n_points = len(titles)
    print(f"  {len(titles_all)} → {n_points} after dedup")

    # ── Build PCA tree (deep, for adaptive cutting) ──────────────────────
    print(f"\nBuilding PCA tree (depth={args.max_depth})...")
    t0 = time.time()
    all_idx = np.arange(n_points)
    tree = _build_pca_tree_with_margins(embeddings, all_idx, args.max_depth)
    print(f"  Built in {time.time() - t0:.1f}s")

    # ── Adaptive cut: stop splitting at dense regions ────────────────────
    print(f"\n=== Adaptive cut (sim_threshold={args.sim_threshold}) ===")
    labels_adaptive, cluster_info = cut_tree_adaptive(
        tree, embeddings, sim_threshold=args.sim_threshold)

    # ── Standard UMAP — baseline ─────────────────────────────────────────
    dyf_k = suggest_n_neighbors(embeddings)
    print(f"\n=== Standard UMAP (n_neighbors={dyf_k}) ===")
    t0 = time.time()
    coords_standard = run_umap(embeddings, n_neighbors=dyf_k)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Bridged kNN using adaptive clusters → UMAP ───────────────────────
    # Use adaptive labels to define within-cluster candidates
    n_adaptive = len(set(labels_adaptive))
    print(f"\n=== Bridged PCA tree kNN (adaptive {n_adaptive} clusters, "
          f"k={args.k}, budget={args.bridge_budget}, bpp={args.bridges_per_point}) ===")

    knn_indices, knn_dists = build_bridged_pca_tree_knn(
        embeddings, tree, labels_adaptive, k=args.k,
        bridge_budget=args.bridge_budget,
        bridges_per_point=args.bridges_per_point,
    )

    # Refine via nn-descent: check neighbors-of-neighbors
    print(f"\n  Refining kNN via nn-descent...")
    t0 = time.time()
    knn_indices, knn_dists = nn_descent_refine(
        knn_indices, knn_dists, embeddings, n_iters=args.nn_descent_iters)
    print(f"  nn-descent done in {time.time() - t0:.1f}s")

    print(f"\n  Running UMAP with precomputed kNN...")
    t0 = time.time()
    coords_bridged = run_umap(embeddings,
                               precomputed_knn=(knn_indices, knn_dists))
    print(f"  Done in {time.time() - t0:.1f}s")

    # Use adaptive labels for evaluation and visualization
    labels = labels_adaptive
    n_actual = len(set(labels))
    sizes = [int(np.sum(labels == c)) for c in set(labels)]
    print(f"\nAdaptive clusters: {n_actual}, sizes: {min(sizes)}-{max(sizes)}")

    # ── Metrics ──────────────────────────────────────────────────────────
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components as cc
    from scipy.sparse import lil_matrix as lil

    def compute_spatial_metrics(coords, labels, embs, name, k_purity=15):
        """Compute metrics that capture spatial coherence and fragmentation."""
        unique_labels = sorted(set(labels))
        n = len(labels)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n = embs / np.maximum(norms, 1e-10)

        # 1. kNN purity in 2D: fraction of spatial neighbors sharing cluster label
        nn = NearestNeighbors(n_neighbors=k_purity + 1)
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        purities = []
        for i in range(n):
            neighbors = indices[i, 1:]
            same = np.sum(labels[neighbors] == labels[i])
            purities.append(same / k_purity)
        mean_purity = np.mean(purities)

        # 2. Cluster fragmentation: connected components per cluster
        #    Threshold = median nearest-neighbor distance * 3
        nn1 = NearestNeighbors(n_neighbors=2)
        nn1.fit(coords)
        dists1, _ = nn1.kneighbors(coords)
        median_nn_dist = np.median(dists1[:, 1])
        threshold = median_nn_dist * 3

        frag_counts = []
        total_fragments = 0
        for c in unique_labels:
            mask = labels == c
            cluster_coords = coords[mask]
            nc = cluster_coords.shape[0]
            if nc < 2:
                frag_counts.append(1)
                total_fragments += 1
                continue
            nn_c = NearestNeighbors(radius=threshold)
            nn_c.fit(cluster_coords)
            adj_c = nn_c.radius_neighbors_graph(cluster_coords, mode='connectivity')
            n_comp, _ = cc(adj_c, directed=False)
            frag_counts.append(n_comp)
            total_fragments += n_comp

        mean_fragments = np.mean(frag_counts)
        max_fragments = max(frag_counts)
        pct_single = sum(1 for f in frag_counts if f == 1) / len(frag_counts) * 100

        # 3. Intra-cluster cosine similarity (high-D)
        #    Mean pairwise cosine similarity within each cluster
        intra_sims = []
        for c in unique_labels:
            mask = labels == c
            cluster_embs = embs_n[mask]
            nc = cluster_embs.shape[0]
            if nc < 2:
                continue
            # Sample for large clusters
            if nc > 200:
                rng = np.random.default_rng(42 + c)
                idx = rng.choice(nc, 200, replace=False)
                cluster_embs = cluster_embs[idx]
                nc = 200
            sim_matrix = cluster_embs @ cluster_embs.T
            # Mean of upper triangle (excluding diagonal)
            mask_tri = np.triu(np.ones((nc, nc), dtype=bool), k=1)
            intra_sims.append(float(sim_matrix[mask_tri].mean()))
        mean_intra_sim = np.mean(intra_sims)
        min_intra_sim = np.min(intra_sims)
        max_intra_sim = np.max(intra_sims)

        # 4. Inter-cluster cosine similarity (high-D)
        #    Mean pairwise cosine similarity between cluster centroids
        centroids_hd = []
        for c in unique_labels:
            centroids_hd.append(embs_n[labels == c].mean(axis=0))
        centroids_hd = np.array(centroids_hd)
        # Renormalize centroids
        centroids_hd = centroids_hd / np.maximum(
            np.linalg.norm(centroids_hd, axis=1, keepdims=True), 1e-10)
        inter_sim_matrix = centroids_hd @ centroids_hd.T
        nc_centroids = len(centroids_hd)
        inter_mask = np.triu(np.ones((nc_centroids, nc_centroids), dtype=bool), k=1)
        mean_inter_sim = float(inter_sim_matrix[inter_mask].mean())

        # 5. Silhouette (2D, for reference)
        from sklearn.metrics import silhouette_score
        silh = silhouette_score(coords, labels,
                                sample_size=min(5000, n))

        print(f"\n  {name}:")
        print(f"    Silhouette (2D):        {silh:>7.3f}")
        print(f"    kNN purity (k={k_purity}):    {mean_purity:>7.3f}  "
              f"(1.0 = all spatial neighbors same cluster)")
        print(f"    Cluster fragments:      {mean_fragments:>7.1f} avg, "
              f"{max_fragments} max  ({pct_single:.0f}% single-blob)")
        print(f"    Total fragments:        {total_fragments:>5d}  "
              f"(ideal: {len(unique_labels)})")
        print(f"    Intra-cluster sim:      {mean_intra_sim:>7.3f}  "
              f"(min={min_intra_sim:.3f}, max={max_intra_sim:.3f})")
        print(f"    Inter-cluster sim:      {mean_inter_sim:>7.3f}  "
              f"(lower = better separated)")
        print(f"    Sim gap (intra-inter):  {mean_intra_sim - mean_inter_sim:>7.3f}  "
              f"(positive = clusters are coherent in high-D)")

        return {
            'silh': silh, 'purity': mean_purity,
            'mean_frag': mean_fragments, 'max_frag': max_fragments,
            'total_frag': total_fragments, 'pct_single': pct_single,
            'intra_sim': mean_intra_sim, 'inter_sim': mean_inter_sim,
        }

    # ── kNN recall: how many true neighbors did the PCA tree kNN find? ──
    print(f"\n=== kNN Recall (brute-force ground truth, k={args.k}) ===")
    from sklearn.neighbors import NearestNeighbors

    # Compute true kNN in high-D cosine space
    nn_true = NearestNeighbors(n_neighbors=args.k + 1, metric='cosine')
    nn_true.fit(embeddings)
    true_dists, true_indices = nn_true.kneighbors(embeddings)
    # true_indices[:, 0] is self — skip it
    true_sets = [set(true_indices[i, 1:]) for i in range(n_points)]

    # Our precomputed kNN (skip self at position 0)
    our_sets = [set(int(x) for x in knn_indices[i, 1:]) for i in range(n_points)]

    recalls = [len(true_sets[i] & our_sets[i]) / args.k for i in range(n_points)]
    mean_recall = np.mean(recalls)
    median_recall = np.median(recalls)
    min_recall = np.min(recalls)
    p10_recall = np.percentile(recalls, 10)

    print(f"  Mean recall:    {mean_recall:.3f}  ({mean_recall*args.k:.1f}/{args.k} true neighbors found)")
    print(f"  Median recall:  {median_recall:.3f}")
    print(f"  10th pct:       {p10_recall:.3f}  (worst 10% of points)")
    print(f"  Min recall:     {min_recall:.3f}")

    # Recall by cluster size
    cluster_sizes_map = {}
    for c in set(labels):
        cluster_sizes_map[c] = int(np.sum(labels == c))

    small_recalls = [recalls[i] for i in range(n_points)
                     if cluster_sizes_map[labels[i]] < 50]
    large_recalls = [recalls[i] for i in range(n_points)
                     if cluster_sizes_map[labels[i]] >= 200]
    if small_recalls:
        print(f"  Small clusters (<50 pts): mean recall {np.mean(small_recalls):.3f}")
    if large_recalls:
        print(f"  Large clusters (>=200 pts): mean recall {np.mean(large_recalls):.3f}")

    print(f"\n=== Spatial Metrics ({n_actual} PCA tree clusters) ===")
    for name, coords in [("Standard UMAP", coords_standard),
                          ("Bridged kNN UMAP", coords_bridged)]:
        compute_spatial_metrics(coords, labels, embeddings, name)

    # ── Generate figures ─────────────────────────────────────────────────
    micro = cut_tree_to_labels(tree, args.max_depth, n_points, 200)
    colors = build_colors_golden_ratio(labels, micro)

    output_paths = []
    for name, coords in [("standard_umap", coords_standard),
                          ("bridged_knn_umap", coords_bridged)]:
        names, centroids = labels_to_names_centroids(coords, titles, labels)
        fig = build_figure(
            coords, titles, labels, colors, names, centroids,
            f"{name} — {n_actual} PCA tree clusters ({min(sizes)}-{max(sizes)} pts)",
        )
        path = str(outdir / f"rog_2d_{name}.html")
        fig.write_html(path, config={"scrollZoom": True})
        print(f"  Wrote {path}")
        output_paths.append(path)

    for p in output_paths:
        subprocess.run(["open", p])


if __name__ == "__main__":
    main()
