"""Agglomerate DYF tree leaves into semantic buckets.

Walks the tree to find leaf nodes, computes per-leaf embedding centroids,
then uses complete linkage or Louvain community detection to merge leaves
into agglomerated clusters.  After initial assignment, iteratively reassigns
individual points to their nearest bucket centroid to clean up impure leaves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class LouvainHierarchy(TypedDict):
    """Return type for compute_louvain_hierarchy()."""

    point_labels: np.ndarray  # int32 (N,)
    leaf_to_community: dict[int, int]
    community_sizes: dict[int, int]
    Z: np.ndarray  # (k-1, 4) float64
    unique_community_ids: list[int]
    leaf_item_map: dict[int, list[int]]
    natural_k: int
    resolution: float
    centroid_dist: np.ndarray  # float32 (N,)
    nearest_other_dist: np.ndarray  # float32 (N,)
    community_cohesion: dict[int, float]  # mean centroid_dist per community (angular spread)
    community_inflow: dict[int, float]  # share of each community moved in by the re-cut
    community_embedding_centroids: np.ndarray  # float32 (k, D)


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
    leaves = [n for n in tree if n["is_leaf"] and n["batch_index"] >= 0]

    if len(leaves) < 2:
        return None

    dim = idx.embedding_dim
    is_pq = idx.is_pq
    if is_pq:
        idx._load_pq_codebook()

    leaf_centroids = []
    leaf_point_indices = []

    for leaf in leaves:
        batch = idx.get_leaf(leaf["batch_index"])
        item_ids = batch.column("item_index").to_numpy()
        emb_col = batch.column("embedding")
        flat = emb_col.values.to_numpy()
        n_rows = len(emb_col)

        if is_pq:
            meta = idx._get_metadata()
            m = int(meta["pq_n_subquantizers"])
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
        item_leaf_map[item_ids[valid]] = leaf["node_id"]
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
        bucket_centroids = np.zeros((n_buckets, embeddings.shape[1]), dtype=np.float32)
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
        new_labels = np.array([unique_groups[bi] for bi in best_bucket_idx], dtype=np.int32)

        changed = int((new_labels != point_labels).sum())
        point_labels = new_labels
        if changed == 0:
            break

    logger.info(f"    Point reassignment: {_iter + 1} iterations, {changed} changed in last round")
    return point_labels


def _unit_rows(a):
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, 1e-10)


def _group_centroids(point_labels, emb_normed, k, fallback=None):
    """Unit mean direction per group id ``0..k-1``.

    A group with no points keeps its ``fallback`` row (its previous centroid) when one is
    given, else stays zero. Returns ``(centroids, empty_group_ids)``.
    """
    centroids = np.zeros((k, emb_normed.shape[1]), dtype=np.float32)
    np.add.at(centroids, point_labels, emb_normed)
    counts = np.bincount(point_labels, minlength=k)
    nonempty = counts > 0
    centroids[nonempty] = _unit_rows(centroids[nonempty])
    if fallback is not None:
        centroids[~nonempty] = fallback[~nonempty]
    return centroids, np.flatnonzero(~nonempty).astype(np.int32)


def _assign_from_centroids(embeddings, init_centroids, max_iter=5):
    """Nearest-centroid rounds (cosine) started from given centroids; group id = row index.

    Same loop as :func:`_reassign_points`, but the starting groups come from the caller, so
    ids carry over between runs. A group that loses every point keeps its starting centroid.
    """
    emb_normed = _unit_rows(embeddings)
    start = _unit_rows(init_centroids)
    k = len(start)
    centroids = start
    point_labels = np.argmax(emb_normed @ centroids.T, axis=1).astype(np.int32)
    changed = 0
    _iter = 0
    for _iter in range(max_iter):
        centroids, _ = _group_centroids(point_labels, emb_normed, k, fallback=start)
        new_labels = np.argmax(emb_normed @ centroids.T, axis=1).astype(np.int32)
        changed = int((new_labels != point_labels).sum())
        point_labels = new_labels
        if changed == 0:
            break
    centroids, empty = _group_centroids(point_labels, emb_normed, k, fallback=start)
    logger.info(f"    Warm-start assignment: {_iter + 1} iterations, {changed} changed in last round")
    return point_labels, centroids, empty


def _check_init_centroids(init_centroids, embeddings, n_groups):
    init = np.asarray(init_centroids, dtype=np.float32)
    if init.ndim != 2 or init.shape[0] < 1:
        raise ValueError(f"init_centroids must be a (k, D) array with k >= 1, got shape {init.shape}")
    if init.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"init_centroids have dimension {init.shape[1]} but embeddings have {embeddings.shape[1]}; "
            "they must come from the same embedding space"
        )
    if not np.isfinite(init).all():
        raise ValueError("init_centroids contain NaN or inf")
    if (np.linalg.norm(init, axis=1) < 1e-10).any():
        raise ValueError("init_centroids contain a zero row, which has no direction to assign to")
    if n_groups is not None and n_groups != len(init):
        raise ValueError(
            f"n_groups={n_groups} conflicts with {len(init)} init_centroids; with init_centroids the "
            "number of groups is len(init_centroids), so leave n_groups unset"
        )
    return init


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
        lsh_label_data.append(
            {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]) if ndim >= 3 else 0.0,
                "text": f"Bucket {gid}",
                "size": int(mask.sum()),
                "cid": int(gid),
                "leaf_cids": [int(gid)],
            }
        )
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
    from scipy.cluster.hierarchy import fcluster, linkage

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
    Z = linkage(centroids_normed, method="complete")
    merge_labels = fcluster(Z, max_k, criterion="maxclust")  # 1-based

    # Remap point labels: old_gid → merge group (0-based)
    remap = {gid: int(merge_labels[id_to_idx[gid]]) - 1 for gid in unique_ids}
    merged = np.array([remap[int(g)] for g in point_labels], dtype=np.int32)

    # Reassign points to nearest merged centroid
    merged = _reassign_points(merged, embeddings)

    n_merged = len(set(merged.tolist()))
    logger.info(f"    Merged {current_k} → {n_merged} communities (max_k={max_k})")
    return merged


def _compute_community_linkage(point_labels, embeddings):
    """Compute linkage matrix Z over community centroids.

    Returns the scipy linkage matrix Z where each row is
    [community_a, community_b, distance, merged_size].
    Community IDs in Z are 0-based indices into the sorted unique community list.

    Args:
        point_labels: int32 array (N,) of community IDs.
        embeddings: (N, D) embedding matrix (float32).

    Returns:
        Tuple of (Z, unique_ids, centroids) where Z is (k-1, 4) float64
        array, unique_ids is the sorted list of original community IDs,
        and centroids is (k, D) float32 array of per-community mean
        embeddings.
    """
    from scipy.cluster.hierarchy import linkage

    unique_ids = sorted(set(point_labels.tolist()))
    k = len(unique_ids)
    dim = embeddings.shape[1]

    centroids = np.zeros((k, dim), dtype=np.float32)
    id_to_idx = {gid: i for i, gid in enumerate(unique_ids)}
    for gid in unique_ids:
        mask = point_labels == gid
        centroids[id_to_idx[gid]] = embeddings[mask].mean(axis=0)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    Z = linkage(centroids_normed, method="complete")
    return Z, unique_ids, centroids


@dataclass(frozen=True)
class LeafGraphSpectrum:
    """The bottom of the normalised-Laplacian spectrum of the leaf graph Louvain ran on.

    A read-out, not a decision: it says which kind of graph the community stage was handed.
    Modularity is the right objective for a graph whose natural cuts are all of similar
    strength (a mixture of blobs); it is the wrong one for a graph that is disconnected, or one
    weak cut away from it (rings, arcs, well-separated groups), where it chops the pieces into
    similar-size communities instead of cutting at the weak edge.

    Measured 2026-09-27 on the gallery shapes and the labelled bench (KNOWN_ISSUES #11):

    ============  =========  ===============  =============================================
    dataset       lambda_2   max ratio        what worked
    ============  =========  ===============  =============================================
    moons         0.00013    10x (i=2)        spectral bisection 0.98 NMI; Louvain 0.42
    circles       0 (x2)     inf              components 1.00; Louvain 0.39
    5 blobs       ~0 (x4)    15-100x (i=4)    spectral k=4-5 ~0.75; Louvain 0.51-0.58
    digits        0.011      1.8              Louvain 0.71 > spectral 0.69
    MNIST PCA-50  0.025      1.3              Louvain 0.66 > spectral 0.59
    ============  =========  ===============  =============================================

    Attributes:
        eigenvalues: the smallest ``n_eigen`` eigenvalues of the symmetric normalised
            Laplacian, ascending, clipped at 0. ``eigenvalues[0]`` is always ~0.
        lambda2: algebraic connectivity — how weak the weakest cut is. ~1e-4 means one
            edge away from disconnected; ~1e-2 means no cut is much weaker than another.
        n_components: eigenvalues that are zero to numerical precision — the connected
            components of the graph. A graph one weak edge from splitting still has
            ``n_components == 1``; that case shows up as a tiny ``lambda2`` instead.
        k_gap: the eigenvalue count before the largest *multiplicative* jump
            ``eigenvalues[i] / eigenvalues[i-1]``, searched from ``max(2, n_components)``.
            The additive eigengap is the wrong decoder for manifolds (their spectrum grows
            smoothly, so the largest additive gap sits far up — it picked k=14 on moons).
        max_ratio: that largest jump. Below ~2 there is no natural k in the spectrum.
        regime: ``"connectivity"`` when ``n_components > 1`` or (``lambda2`` below the
            threshold and ``max_ratio`` above it); ``"blob"`` otherwise; ``"degenerate"``
            for graphs too small to say anything. Thresholds are the measured ones above,
            not universal constants — they sit between two regimes ~100x apart at k=5-10.
        intrinsic_dim: Weyl-law estimate of the manifold dimension the leaves sample. On a
            d-dimensional manifold the Laplacian eigenvalues grow like ``k ** (2/d)``, so the
            log-log slope of the low spectrum (after the component zeros) estimates ``2/d``.
            Measured: synthetic ring 0.9, path 0.8, 2-D blob 1.5 (noisy from 11 eigenvalues),
            the brain's largest neuron cluster 2.9, its cell-cycle and endothelium clusters 1.8 —
            reproducing the covariance-based classes of the diagnostic stack from the graph
            alone. NaN when there are too few eigenvalues to fit.

    A ring detector was tried here and removed. A cycle graph's eigenvalues come in equal
    pairs, so the relative gap inside (λ₂,λ₃), (λ₄,λ₅), (λ₆,λ₇) separates a synthetic ring
    (0.14) from a path (0.48) cleanly. On the 1.29M-cell brain, against a within-cluster
    column-shuffle null, the known cell-cycle cluster paired *less* than its null (0.198 vs
    0.130 ± 0.031) and the score ranged 0.08–0.43 across the 35 clusters with no relation to
    which ones carry cycles; no cluster is near 1-D, the only regime where pairing means
    anything. For what it is worth, persistent homology on the whole brain's bucket centroids
    did not find a clean cell-cycle ring in this dataset either (E18 cycling cells span
    lineages); the cycle it did confirm under the same null, at z=+12.8, is the vascular
    state-cycle spanning the endothelium and pericyte clusters — a feature that crosses cluster
    boundaries, which a per-cluster spectrum cannot see by construction. KNOWN_ISSUES #11.
    """

    eigenvalues: np.ndarray
    lambda2: float
    n_components: int
    k_gap: int
    max_ratio: float
    regime: str
    intrinsic_dim: float = float("nan")

    def summary(self) -> str:
        if self.regime == "degenerate":
            return "leaf graph too small for a spectrum"
        dim = f", ~{self.intrinsic_dim:.1f}-D" if np.isfinite(self.intrinsic_dim) else ""
        return (
            f"leaf graph: {self.regime} regime — lambda_2={self.lambda2:.2e}, "
            f"{self.n_components} component(s), largest spectral jump {self.max_ratio:.1f}x "
            f"after {self.k_gap} eigenvalue(s){dim}"
        )


def leaf_graph_spectrum(
    edges,
    n_leaves: int,
    *,
    n_eigen: int = 16,
    ratio_threshold: float = 5.0,
    lambda2_threshold: float = 1e-3,
) -> LeafGraphSpectrum:
    """Compute :class:`LeafGraphSpectrum` for a weighted leaf graph.

    Args:
        edges: iterable of ``(src, dst, weight)`` — the same edge list handed to
            ``dyf_rs.louvain_communities``. Direction is ignored (symmetrised by max).
        n_leaves: number of nodes.
        n_eigen: how many of the smallest eigenvalues to compute (dense below 2,000 nodes,
            ``scipy.sparse.linalg.eigsh`` above).
        ratio_threshold, lambda2_threshold: the regime cut-offs (see the class docstring).

    Cost: milliseconds at a few hundred leaves, ~1 s at a few thousand.
    """
    import scipy.sparse as sp

    if n_leaves < 1:
        raise ValueError(f"n_leaves must be positive, got {n_leaves}")
    edges = list(edges)
    m = min(n_eigen, n_leaves)
    degenerate = LeafGraphSpectrum(
        eigenvalues=np.zeros(min(m, 1)), lambda2=0.0, n_components=1, k_gap=1, max_ratio=0.0, regime="degenerate"
    )
    if n_leaves < 3 or not edges:
        return degenerate

    src = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    dst = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    w = np.fromiter((float(e[2]) for e in edges), dtype=np.float64, count=len(edges))
    W = sp.coo_matrix((w, (src, dst)), shape=(n_leaves, n_leaves)).tocsr()
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    if n_leaves <= 2000:
        lsym = np.eye(n_leaves) - (d_inv_sqrt[:, None] * W.toarray() * d_inv_sqrt[None, :])
        vals = np.linalg.eigvalsh(lsym)[:m]
    else:
        from scipy.sparse.linalg import eigsh

        lsym = sp.identity(n_leaves, format="csr") - sp.diags(d_inv_sqrt) @ W @ sp.diags(d_inv_sqrt)
        vals = np.sort(eigsh(lsym, k=m, which="SA", return_eigenvectors=False))
    vals = np.clip(vals, 0.0, None)

    # Exact components: eigenvalues that are zero to numerical precision (the normalised
    # Laplacian's spectrum is O(1), so an absolute tolerance is meaningful). Near-disconnection
    # is reported through lambda2 and the ratio, not folded into this count.
    n_components = max(1, int((vals < 1e-6).sum()))
    # Ratio search floor: tiny eigenvalues are numerically noisy, so a ratio between two of
    # them means nothing. Relative to the top of the window so it scales with the graph.
    floor = max(1e-9, 1e-3 * float(vals[-1]))
    lambda2 = float(vals[1])
    # multiplicative jumps, 1-based: ratio_i = lambda_{i+1} / lambda_i for i >= max(2, n_components)
    start = max(2, n_components)
    k_gap, max_ratio = n_components, 0.0
    for i in range(start, m):  # i is 1-based index of the eigenvalue before the jump
        ratio = float(vals[i] / max(vals[i - 1], floor))
        if ratio > max_ratio:
            max_ratio, k_gap = ratio, i
    if n_components > 1 and max_ratio < ratio_threshold:
        k_gap = n_components
    if n_components > 1 or (lambda2 < lambda2_threshold and max_ratio >= ratio_threshold):
        regime = "connectivity"
    else:
        regime = "blob"

    # Weyl: lambda_k ~ k^(2/d). Fit log lambda against log k over the eigenvalues after the
    # component zeros (k re-indexed from 1 there), using at most the first 11 of them.
    tail = vals[n_components : min(n_components + 11, m)]
    tail = tail[tail > floor]
    intrinsic_dim = float("nan")
    if len(tail) >= 4:
        slope = float(np.polyfit(np.log(np.arange(1, len(tail) + 1)), np.log(tail), 1)[0])
        if slope > 0:
            intrinsic_dim = 2.0 / slope

    return LeafGraphSpectrum(
        eigenvalues=vals,
        lambda2=lambda2,
        n_components=n_components,
        k_gap=int(k_gap),
        max_ratio=max_ratio,
        regime=regime,
        intrinsic_dim=intrinsic_dim,
    )


def _run_louvain_on_centroids(centroids_normed, k, resolution, similarity_threshold):
    """Run Louvain community detection on L2-normalized leaf centroids.

    Tries the Rust implementation first (faster, weighted, deterministic),
    falling back to NetworkX Louvain if dyf_rs is not available.

    Args:
        centroids_normed: (L, D) float32 array of L2-normalized leaf centroids.
        k: Number of nearest neighbors per centroid.
        resolution: Louvain resolution parameter.
        similarity_threshold: Minimum cosine similarity to keep a KNN edge
            (only used by the NetworkX fallback).

    Returns:
        Tuple of (leaf_labels, n_communities, spectrum) where leaf_labels is an int32
        array (L,), n_communities is the count of distinct communities, and spectrum is
        the :class:`LeafGraphSpectrum` of the graph Louvain ran on (None on the NetworkX
        fallback path).
    """
    spectrum = None
    try:
        from dyf_rs import build_knn_graph, louvain_communities

        from ._arrays import ensure_f32

        # Build the graph once and hand the same edges to Louvain and to the spectrum, so
        # the read-out describes exactly the graph the communities came from.
        adjacency = build_knn_graph(ensure_f32(centroids_normed, "centroids_normed"), k=k)
        edges = [(i, int(j), float(wt)) for i, nbrs in enumerate(adjacency) for j, wt in nbrs]
        labels_arr, n_communities = louvain_communities(len(centroids_normed), edges, resolution=resolution)
        leaf_labels = np.asarray(labels_arr).astype(np.int32)
        spectrum = leaf_graph_spectrum(edges, len(centroids_normed))
        logger.info(
            f"    Louvain (Rust) found {n_communities} communities "
            f"from {len(centroids_normed)} leaves (k={k}, res={resolution}); {spectrum.summary()}"
        )
        if spectrum.regime == "connectivity":
            logger.warning(
                "    Leaf graph is disconnected or one weak cut from it (lambda_2=%.2e, %d component(s), "
                "jump %.1fx after %d eigenvalues). Modularity tends to split such structure into "
                "similar-size pieces; connected components or a spectral cut at k=%d may match it better.",
                spectrum.lambda2,
                spectrum.n_components,
                spectrum.max_ratio,
                spectrum.k_gap,
                spectrum.k_gap,
            )
    except ImportError:
        # Fall back to NetworkX Louvain
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(centroids_normed)
        distances, indices = nn.kneighbors(centroids_normed)

        # Build graph: edge if cosine similarity > threshold
        G = nx.Graph()
        G.add_nodes_from(range(len(centroids_normed)))
        for i in range(len(centroids_normed)):
            for j_pos in range(1, distances.shape[1]):
                j = indices[i, j_pos]
                sim = 1.0 - distances[i, j_pos]
                if sim > similarity_threshold:
                    G.add_edge(i, j, weight=sim)

        # Louvain community detection
        communities = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=42)

        # Map communities → leaf labels (0-based)
        leaf_labels = np.full(len(centroids_normed), -1, dtype=np.int32)
        for comm_id, members in enumerate(communities):
            for leaf_idx in members:
                leaf_labels[leaf_idx] = comm_id

        # Handle isolated nodes
        isolated = leaf_labels == -1
        if isolated.any():
            next_id = leaf_labels.max() + 1
            for i in np.where(isolated)[0]:
                leaf_labels[i] = next_id
                next_id += 1

        n_communities = len(set(leaf_labels.tolist()))
        logger.info(
            f"    Louvain (NetworkX) found {n_communities} communities "
            f"from {len(centroids_normed)} leaves (k={k}, res={resolution})"
        )

    return leaf_labels, n_communities, spectrum


def _compute_point_metrics(point_labels, embeddings, community_centroids_emb, unique_ids):
    """Compute per-point cosine distances and per-community cohesion.

    Args:
        point_labels: int32 array (N,) of community IDs.
        embeddings: (N, D) embedding matrix (float32).
        community_centroids_emb: (k, D) float32 array of community centroids.
        unique_ids: sorted list of unique community IDs.

    Returns:
        Tuple of (centroid_dist, nearest_other_dist, community_cohesion)
        where centroid_dist and nearest_other_dist are float32 arrays (N,)
        and community_cohesion is a dict {community_id: float}.
    """
    # Normalize embeddings and centroids for cosine via dot product
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    emb_normed = embeddings / emb_norms

    cen_norms = np.linalg.norm(community_centroids_emb, axis=1, keepdims=True)
    cen_norms[cen_norms == 0] = 1.0
    cen_normed = community_centroids_emb / cen_norms

    # (N, k) cosine similarity matrix
    sim_matrix = emb_normed @ cen_normed.T

    id_to_idx = {gid: i for i, gid in enumerate(unique_ids)}
    own_idx = np.array([id_to_idx[int(c)] for c in point_labels], dtype=np.int32)

    # Cosine distance to own centroid
    own_sim = sim_matrix[np.arange(len(point_labels)), own_idx]
    centroid_dist = (1.0 - own_sim).astype(np.float32)

    # Nearest OTHER centroid distance
    # Mask own centroid with -inf, then take max similarity
    masked_sim = sim_matrix.copy()
    masked_sim[np.arange(len(point_labels)), own_idx] = -np.inf
    nearest_other_sim = masked_sim.max(axis=1)
    nearest_other_dist = (1.0 - nearest_other_sim).astype(np.float32)

    # Community cohesion: mean centroid_dist per community
    community_cohesion = {}
    for cid in unique_ids:
        mask = point_labels == cid
        community_cohesion[cid] = float(centroid_dist[mask].mean())

    return centroid_dist, nearest_other_dist, community_cohesion


def compute_louvain_hierarchy(
    idx, coords, embeddings, leaf_k=10, similarity_threshold=0.5, resolution=1.0
) -> LouvainHierarchy | None:
    """Compute Louvain communities with dendrogram for continuous cluster slider.

    Returns the three artifacts needed for the dendrogram-based slider:
    1. leaf_communities: mapping from Louvain community → leaf indices
    2. dendrogram Z: scipy linkage matrix over community centroids
    3. leaf_item_map: mapping from tree leaf → item indices

    Args:
        idx: An open ``LazyIndex`` handle.
        coords: (N, 2-or-3) UMAP coordinates for every item.
        embeddings: (N, D) embedding matrix (float32).
        leaf_k: Number of nearest neighbors per leaf centroid (default 10).
        similarity_threshold: Minimum cosine similarity to keep an edge
            (default 0.5).
        resolution: Louvain resolution parameter (default 1.0).

    Returns:
        Dict with keys:
            ``point_labels``: int32 array (N,) of community IDs (0-based). These are the
                labels *after* ``_reassign_points``: a Voronoi re-cut of the Louvain
                communities (see ``louvain_cluster_leaves`` and ``KNOWN_ISSUES.md`` #14).
            ``leaf_to_community``: dict {leaf_idx: community_id} — the Louvain stage's
                label per leaf, i.e. where each point sat *before* the re-cut.
            ``community_sizes``: dict {community_id: int}.
            ``Z``: (k-1, 4) linkage matrix.
            ``unique_community_ids``: sorted list of community IDs.
            ``leaf_item_map``: dict {leaf_idx: list of item indices}.
            ``natural_k``: int, number of natural communities.
            ``resolution``: float, Louvain resolution used.
            ``centroid_dist``: float32 (N,), cosine distance from each point to its own
                community centroid.
            ``nearest_other_dist``: float32 (N,), cosine distance to the nearest *other*
                centroid. Per point, neither this nor the margin between the two carries
                a usable confidence signal (measured 2026-11, bench in
                ``benchmarks/cut_audit_2026_11/``); keep them as descriptions.
            ``community_cohesion``: dict {community_id: float}, the mean ``centroid_dist``
                over the community's members — its angular spread, the per-cell term of
                the objective the re-cut minimises. A loose community is either a broad
                region or a Voronoi cell that absorbed its neighbours; ``community_inflow``
                tells the two apart. **Read it as a rank within this index, never against a
                constant**: absolute cosine values do not transfer across corpora. Ranked,
                it picks out the least label-pure quarter of communities at AUC 0.80 on six
                of seven bench sets, and is uninformative on continuum data (MoCap), where a
                loose cell is not a defect.
            ``community_inflow``: dict {community_id: float}, the share of each final
                community whose members carried a *different* Louvain label before the
                re-cut. Near 0 means the cell is the Louvain community it is named after;
                near 1 means a small seed's Voronoi cell swallowed a region (0.94 on one
                150k-point brain community and one 19k droplet community in the bench).
            ``community_embedding_centroids``: float32 (k, D) unit centroids in
                ``unique_community_ids`` order.

        Returns ``None`` when the tree has fewer than two leaves.
    """
    result = _collect_leaf_data(idx)
    if result is None:
        return None

    leaves, leaf_centroids, leaf_point_indices, _tree = result
    n_points = coords.shape[0]

    centroids = np.vstack(leaf_centroids).astype(np.float32)

    # L2-normalize centroids for cosine similarity
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    # KNN on normalized centroids
    k = min(leaf_k, len(centroids_normed) - 1)
    if k < 1:
        k = 1

    leaf_labels, n_communities, _spectrum = _run_louvain_on_centroids(
        centroids_normed, k, resolution, similarity_threshold
    )

    # Map points -> leaf -> community
    point_labels = np.full(n_points, -1, dtype=np.int32)
    for leaf_idx, item_ids in enumerate(leaf_point_indices):
        group_id = int(leaf_labels[leaf_idx])
        valid = item_ids < n_points
        point_labels[item_ids[valid]] = group_id

    # Handle unassigned points
    unassigned = point_labels == -1
    if unassigned.any():
        max_id = point_labels.max() + 1
        point_labels[unassigned] = max_id

    # Point-level reassignment pass (the Voronoi re-cut). Keep the Louvain labels so each
    # final community can report how much of it the re-cut moved in.
    block_labels = point_labels.copy()
    point_labels = _reassign_points(point_labels, embeddings)

    # Build leaf_to_community mapping
    leaf_to_community = {}
    for leaf_idx in range(len(leaf_labels)):
        leaf_to_community[leaf_idx] = int(leaf_labels[leaf_idx])

    # Build leaf_item_map: leaf_idx → list of item indices
    leaf_item_map = {}
    for leaf_idx, item_ids in enumerate(leaf_point_indices):
        leaf_item_map[leaf_idx] = item_ids.tolist()

    # Compute community sizes and re-cut inflow from final point labels
    community_sizes = {}
    community_inflow = {}
    moved = block_labels != point_labels
    for cid in sorted(set(point_labels.tolist())):
        mask = point_labels == cid
        community_sizes[cid] = int(mask.sum())
        community_inflow[cid] = float(moved[mask].mean())

    # Compute linkage dendrogram over community centroids
    Z, unique_ids, community_centroids_emb = _compute_community_linkage(point_labels, embeddings)

    # Per-point metrics: cosine distance to own centroid and nearest other
    centroid_dist, nearest_other_dist, community_cohesion = _compute_point_metrics(
        point_labels, embeddings, community_centroids_emb, unique_ids
    )

    return {
        "point_labels": point_labels,
        "leaf_to_community": leaf_to_community,
        "community_sizes": community_sizes,
        "Z": Z,
        "unique_community_ids": unique_ids,
        "leaf_item_map": leaf_item_map,
        "natural_k": n_communities,
        "resolution": resolution,
        "centroid_dist": centroid_dist,
        "nearest_other_dist": nearest_other_dist,
        "community_cohesion": community_cohesion,
        "community_inflow": community_inflow,
        "community_embedding_centroids": community_centroids_emb,
    }


@dataclass
class LeafGroupingResult:
    """Result of grouping a DYF tree's leaves — :func:`agglomerate_tree_leaves` and
    :func:`louvain_cluster_leaves` both return this.

    Replaces a bare 5-tuple that both functions returned identically, documented as
    "Same tuple as ``agglomerate_tree_leaves``" — a struct in everything but name.

    ``ok`` is the fact the tuple could not express. On the too-few-leaves path the old
    code returned ``(None, {}, [], None, tree)``: four sentinel values that a caller had
    to decode by testing position 0 for ``None``. The rest of this package raises
    ``ValueError`` for unusable input, so the sentinel was also inconsistent — but
    raising here would break callers that legitimately handle small trees, so the
    condition is named instead.

    Unpacks as the original 5-tuple, so existing callers are unaffected.

    Note there is deliberately no ``__len__``: the only honest meanings would be "5"
    (the unpacking arity, which is useless) or the group count, and ``SearchResult``
    already demonstrated how a hard-coded arity becomes a plausible wrong number. Use
    :attr:`n_groups`.
    """

    point_labels: np.ndarray | None
    names: dict
    label_data: list
    item_leaf_map: np.ndarray | None
    tree: object
    spectrum: LeafGraphSpectrum | None = None
    """Spectrum of the leaf graph the communities came from (:func:`louvain_cluster_leaves`
    only; None for :func:`agglomerate_tree_leaves` and the too-few-leaves case). Not part of
    the 5-tuple unpacking."""
    centroids: np.ndarray | None = None
    """(k, D) unit centroid per group, row ``i`` = group ``i`` (:func:`agglomerate_tree_leaves`
    only). Pass it as ``init_centroids`` to the next run to keep group ids. Not part of the
    5-tuple unpacking."""
    empty_groups: np.ndarray | None = None
    """Ids of groups with no points this run (warm start only; their centroid row is the one
    passed in, so the id stays available). Empty array when every group has points."""
    mean_cosine: float | None = None
    """Mean cosine between each point and its group centroid. Compare it with the previous
    run's: a sharp drop after a warm start means the old centroids no longer fit the data
    (new embedding model, unaligned frame, or a real regime change), even though ids carried
    over."""

    def _as_tuple(self):
        return (self.point_labels, self.names, self.label_data, self.item_leaf_map, self.tree)

    def __iter__(self):
        """Backward-compatible unpacking of the original 5-tuple."""
        return iter(self._as_tuple())

    def __getitem__(self, key):
        """Backward-compatible positional access, e.g. ``result[0]``.

        Integer-only, unlike ``SearchResult.__getitem__``, which is overloaded on key
        type — there is no field mapping here to be ambiguous with. Prefer the named
        attributes; this exists so existing positional callers keep working.
        """
        return self._as_tuple()[key]

    @property
    def ok(self) -> bool:
        """True when grouping actually ran. False means the tree had too few leaves."""
        return self.point_labels is not None

    @property
    def n_groups(self) -> int:
        """Number of distinct groups found; 0 when grouping did not run."""
        if self.point_labels is None:
            return 0
        return int(len(np.unique(self.point_labels)))

    @classmethod
    def insufficient_leaves(cls, tree) -> LeafGroupingResult:
        """The degenerate case, named. Previously ``(None, {}, [], None, tree)``."""
        return cls(point_labels=None, names={}, label_data=[], item_leaf_map=None, tree=tree)

    def summary(self) -> str:
        if not self.ok:
            return "Leaf grouping did not run: tree has fewer than two leaves."
        text = f"{self.n_groups} groups over {len(self.point_labels)} points."
        if self.mean_cosine is not None:
            text += f" Mean cosine to group centroid {self.mean_cosine:.3f}."
        if self.empty_groups is not None and len(self.empty_groups):
            text += f" {len(self.empty_groups)} groups empty: {self.empty_groups.tolist()}."
        if self.spectrum is not None:
            text += f" {self.spectrum.summary()}."
        return text


def agglomerate_tree_leaves(idx, coords, embeddings, n_groups=None, init_centroids=None):
    """Agglomerate DYF tree leaves into ~n_groups using embedding centroids.

    Walks the tree to find leaf nodes, computes per-leaf embedding centroids,
    then uses complete linkage to merge leaves into n_groups agglomerated
    clusters.  After initial assignment, reassigns individual points to their
    nearest bucket centroid to clean up impure leaves and bad merges.

    The reassignment is spherical k-means, so the result is one of many
    near-equivalent solutions and its group ids are arbitrary: rerun on slightly
    different data (an extended index, the next time window) and the groups
    reshuffle. Pass the previous result's ``centroids`` as ``init_centroids`` to
    start from them instead of from the leaf merge. Group ``i`` is then the group
    that grew from row ``i``, so ids carry over and the solution stays near the
    old one. On quarterly stock-correlation windows this kept 75% of groups
    (Jaccard >= 0.5) against 41% for refitting, at the same within-group
    co-movement (sec10quant, 2026-11-01).

    Warm-started ids are stable even when the data under them is not: a group
    can keep its id while its members turn over, and a structural break leaves
    the ids in place. Watch ``mean_cosine`` against the previous run, and use a
    separate break check where that matters.

    Args:
        idx: An open ``LazyIndex`` handle (used to read tree structure and
            leaf batches).
        coords: (N, 2-or-3) UMAP coordinates for every item.
        embeddings: (N, D) embedding matrix (float32).
        n_groups: Target number of agglomerated buckets (default 50). With
            ``init_centroids`` the count is ``len(init_centroids)``; leave this unset.
        init_centroids: Optional (k, D) array, usually a previous result's
            ``centroids``, in the same embedding space as ``embeddings``. Skips the
            leaf merge; group ids are row indices; a group that gets no points
            keeps its row and is listed in ``empty_groups``. The tree is still read
            for ``item_leaf_map``.

    Raises:
        ValueError: ``init_centroids`` has the wrong shape or dimension, contains
            non-finite values or a zero row, or disagrees with ``n_groups``.

    Returns:
        :class:`LeafGroupingResult`, ready for ``multi_level_data``. Unpacks as the
        original 5-tuple ``(point_labels, names, label_data, item_leaf_map, tree)``.

        *  ``point_labels`` – int32 array (N,) of bucket ids (0-based).
        *  ``names`` – ``{cid: "Bucket <cid>"}`` placeholder names.
        *  ``label_data`` – list of dicts with centroid x/y/z, size, cid.
        *  ``item_leaf_map`` – int32 array (N,) mapping each item to its
           tree leaf ``node_id`` (before agglomeration).
        *  ``tree`` – raw tree node list from ``idx.get_tree_structure()``.

        Also carries ``centroids`` (row ``i`` = group ``i``), ``empty_groups`` and
        ``mean_cosine``. Without ``init_centroids``, ids are relabelled to
        ``0..n_groups-1`` after reassignment, so ``centroids`` has no gaps.

        When the tree has fewer than two leaves, ``.ok`` is False and the array fields
        are None — check ``.ok`` rather than testing ``point_labels is None``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    if init_centroids is not None:
        init_centroids = _check_init_centroids(init_centroids, embeddings, n_groups)
    elif n_groups is None:
        n_groups = 50

    result = _collect_leaf_data(idx)
    if result is None:
        tree = idx.get_tree_structure()
        return LeafGroupingResult.insufficient_leaves(tree)

    leaves, leaf_centroids, leaf_point_indices, tree = result
    n_points = coords.shape[0]

    item_leaf_map = _build_item_leaf_map(leaves, leaf_point_indices, n_points)

    if init_centroids is not None:
        point_labels, group_centroids, empty = _assign_from_centroids(embeddings, init_centroids)
        return _grouping_result(point_labels, group_centroids, empty, embeddings, coords, item_leaf_map, tree)

    centroids = np.vstack(leaf_centroids).astype(np.float32)

    # L2-normalize centroids for cosine-distance complete linkage
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids_normed = centroids / norms

    # Agglomerate -- complete linkage refuses merges where the most
    # dissimilar pair across two groups exceeds the threshold, which
    # prevents merging semantically distinct subgroups.
    actual_groups = min(n_groups, len(centroids_normed))
    Z = linkage(centroids_normed, method="complete")
    agg_labels = fcluster(Z, actual_groups, criterion="maxclust")  # 1-based

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

    # Reassignment can empty a group; relabel to 0..k-1 so centroid rows have no gaps
    point_labels = np.unique(point_labels, return_inverse=True)[1].astype(np.int32)
    k = int(point_labels.max()) + 1
    group_centroids, empty = _group_centroids(point_labels, _unit_rows(embeddings), k)
    return _grouping_result(point_labels, group_centroids, empty, embeddings, coords, item_leaf_map, tree)


def _grouping_result(point_labels, group_centroids, empty, embeddings, coords, item_leaf_map, tree):
    emb_normed = _unit_rows(np.asarray(embeddings, dtype=np.float32))
    mean_cosine = float(np.mean(np.sum(emb_normed * group_centroids[point_labels], axis=1)))
    lsh_names, lsh_label_data = _build_output(point_labels, coords)
    return LeafGroupingResult(
        point_labels=point_labels,
        names=lsh_names,
        label_data=lsh_label_data,
        item_leaf_map=item_leaf_map,
        tree=tree,
        centroids=group_centroids,
        empty_groups=empty,
        mean_cosine=mean_cosine,
    )


def louvain_cluster_leaves(idx, coords, embeddings, leaf_k=10, similarity_threshold=0.5, resolution=1.0):
    """Cluster tree leaves via centroid KNN + Louvain community detection.

    Finds natural communities without requiring a target k.  Builds a KNN
    graph over leaf centroids, filters weak edges by cosine similarity
    threshold, then runs Louvain to discover communities.

    **The partition you get back is not the Louvain partition.** The last step
    (``_reassign_points``) runs up to five rounds of spherical k-means over every
    point, seeded by the Louvain labels, so the result is a Voronoi tessellation of
    the community centroids in cosine space: each point belongs to the community whose
    centroid it is most similar to. The leaf-to-community map is a starting point, not
    the answer. Measured on the seven-dataset bench (``KNOWN_ISSUES.md`` #14): the step
    moves 4–31% of points, makes communities contiguous (the worst community's share of
    neighbours lying outside it falls from 0.6–0.8 to 0.05–0.2), and shifts agreement
    with ground truth by −0.13 (MNIST) to +0.11 (1.3M brain), the sign stable across
    seeds and set by the data — it helps on scRNA PCA scores and hurts on image PCA.
    Across the bench the net ARI effect is about zero; its value is contiguity and a
    reproducible rule for placing new points, not accuracy. There is no flag to skip
    it; ``dyfviz`` stores the post-reassignment label, so changing the default would
    change its files.

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
        The same :class:`LeafGroupingResult` as :func:`agglomerate_tree_leaves` — which
        is now enforced by the shared type rather than asserted in prose. Check ``.ok``
        for the fewer-than-two-leaves case.
    """
    result = _collect_leaf_data(idx)
    if result is None:
        tree = idx.get_tree_structure()
        return LeafGroupingResult.insufficient_leaves(tree)

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

    leaf_labels, _n_communities, spectrum = _run_louvain_on_centroids(
        centroids_normed, k, resolution, similarity_threshold
    )

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
    return LeafGroupingResult(
        point_labels=point_labels,
        names=lsh_names,
        label_data=lsh_label_data,
        item_leaf_map=item_leaf_map,
        tree=tree,
        spectrum=spectrum,
    )
