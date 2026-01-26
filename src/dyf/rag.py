"""
Bridge-based RAG Index for efficient semantic retrieval.

Uses density-based bridge points as anchors for two-tier approximate nearest
neighbor search. Bridge points naturally capture the boundaries between
semantic regions, providing better coverage than centroid-based methods.

Key insight: Bridge points connect multiple LSH buckets and thus provide
natural "crossroads" in embedding space. Using maximally-spread (orthogonal)
bridge points as anchors achieves ~87% recall with ~22% of anchors compared
to using all bridges.

Example:
    >>> from dyf import BridgeIndex
    >>> index = BridgeIndex()
    >>> index.fit(embeddings)
    >>> candidates = index.query(query_embedding, k=10)
    >>> print(index.summary())

Tuning:
    - Speed priority: n_anchors=1000, n_query_anchors=10, expansion_k=200
      → ~87% recall, ~2000 candidates/query
    - Balanced: n_anchors=1500, n_query_anchors=20, expansion_k=400
      → ~97% recall, ~8000 candidates/query
    - Quality priority: n_anchors=1500, n_query_anchors=30, expansion_k=500
      → ~98.6% recall, ~15000 candidates/query

Facet Diversification:
    >>> from dyf import diversify_by_facet
    >>> # Get one result per semantic facet (bucket)
    >>> diverse = diversify_by_facet(query, candidates, embeddings, bucket_ids, k=10)
    >>> # Returns indices of k items from k different buckets

IVF Initialization:
    >>> from dyf import get_kmeans_init
    >>> # Get bridge-seeded centroids for FAISS IVF
    >>> init_centroids = get_kmeans_init(embeddings, nlist=1000)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np

try:
    from dyf_rs import DensityClassifier, BridgeAnalysis
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    # Fall back to Python implementation
    from .classifier import DensityClassifier


@dataclass
class SuperConnectorResult:
    """
    Result of finding super connectors in an embedding space.

    Super connectors are points with high centrality in both global and local
    bridge networks - they connect major semantic regions AND facets within
    dense clusters. They represent ~0.1% of typical corpora but provide
    disproportionate coverage as RAG anchors.

    Attributes:
        indices: Array of super connector point indices
        global_centrality: Global bridge centrality for all points (0 = non-bridge)
        local_centrality: Local (facet) bridge centrality for all points
        quadrant: Category label for each point ('Regular', 'Minor Bridge',
                  'Cross-Domain', 'Domain Specialist', 'Super Connector')
        global_threshold: Threshold used for "high" global centrality
        local_threshold: Threshold used for "high" local centrality
    """
    indices: np.ndarray
    global_centrality: np.ndarray
    local_centrality: np.ndarray
    quadrant: np.ndarray
    global_threshold: float
    local_threshold: float

    def __len__(self) -> int:
        return len(self.indices)

    def summary(self) -> str:
        """Return a summary string of the super connector distribution."""
        n_super = len(self.indices)
        n_cross = (self.quadrant == 'Cross-Domain').sum()
        n_specialist = (self.quadrant == 'Domain Specialist').sum()
        n_minor = (self.quadrant == 'Minor Bridge').sum()
        return (
            f"Super Connectors: {n_super} | Cross-Domain: {n_cross} | "
            f"Domain Specialists: {n_specialist} | Minor Bridges: {n_minor}"
        )


@dataclass
class OrthogonalAnchorResult:
    """
    Result of orthogonal anchor selection.

    Orthogonal anchors are maximally spread points selected using greedy
    farthest-point sampling. Starting from seed points (typically super
    connectors), each subsequent anchor is chosen to maximize minimum
    distance from all previously selected anchors.

    Attributes:
        indices: Array of selected anchor indices
        seed_indices: Initial seed indices (e.g., super connectors)
        candidate_source: Source of candidates ('bridges', 'all', or 'custom')
    """
    indices: np.ndarray
    seed_indices: np.ndarray
    candidate_source: str

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class BridgeIndex:
    """
    Two-tier RAG index using bridge-based anchors for efficient retrieval.

    Instead of indexing all embeddings, BridgeIndex selects a small set of
    "anchor" points that provide good coverage of the embedding space.
    Queries first find nearby anchors, then expand to their neighborhoods.

    The key innovation is using bridge points (which connect multiple LSH
    buckets) as anchors rather than cluster centroids. Bridges naturally
    occur at semantic boundaries and provide better coverage.

    Attributes:
        n_anchors: Number of anchor points to use
        n_query_anchors: Number of anchors to retrieve per query
        expansion_k: Size of neighborhood to expand from each anchor
        global_num_bits: LSH bits for global bucketing (default 12)
        seed: Random seed for reproducibility

    Example:
        >>> index = BridgeIndex(n_anchors=1000)
        >>> index.fit(embeddings)
        >>>
        >>> # Single query
        >>> candidates, scores = index.query(query_vec, k=10)
        >>>
        >>> # Batch queries
        >>> results = index.query_batch(query_vecs, k=10)
    """
    # Configuration
    n_anchors: int = 1000
    n_query_anchors: int = 10
    expansion_k: int = 200
    global_num_bits: int = 12
    facet_num_bits: int = 10
    dense_percentile: float = 75
    min_bucket_size: int = 20
    include_sparse_points: int = 0  # Number of sparse region points to add
    seed: int = 42

    # Fitted state (populated by fit())
    _embeddings: Optional[np.ndarray] = field(default=None, repr=False)
    _anchor_indices: Optional[np.ndarray] = field(default=None, repr=False)
    _anchor_embeddings: Optional[np.ndarray] = field(default=None, repr=False)
    _neighborhoods: Optional[Dict[int, np.ndarray]] = field(default=None, repr=False)
    _super_connectors: Optional[SuperConnectorResult] = field(default=None, repr=False)
    _bridge_indices: Optional[np.ndarray] = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(self, embeddings: np.ndarray, verbose: bool = True) -> 'BridgeIndex':
        """
        Build the bridge index from embeddings.

        Args:
            embeddings: (n, d) array of embedding vectors (will be L2-normalized)
            verbose: Print progress information

        Returns:
            self (for chaining)
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)

        self._embeddings = embeddings
        n_points, dim = embeddings.shape

        if verbose:
            print(f"Building BridgeIndex: {n_points:,} points, dim={dim}")

        # Step 1: Find super connectors
        if verbose:
            print("  Finding super connectors...")
        self._super_connectors = find_super_connectors(
            embeddings,
            global_num_bits=self.global_num_bits,
            facet_num_bits=self.facet_num_bits,
            dense_percentile=self.dense_percentile,
            min_bucket_size=self.min_bucket_size,
            seed=self.seed
        )

        if verbose:
            print(f"    {self._super_connectors.summary()}")

        # Step 2: Get all bridge indices
        clf = DensityClassifier(embedding_dim=dim, num_bits=self.global_num_bits, seed=self.seed)
        clf.fit(embeddings)
        bridge_analysis = clf.analyze_bridges(embeddings)
        self._bridge_indices = np.array(bridge_analysis.bridge_indices)

        if verbose:
            print(f"    Total bridges: {len(self._bridge_indices):,}")

        # Step 3: Select orthogonal anchors
        if verbose:
            print(f"  Selecting {self.n_anchors} orthogonal anchors...")

        # If we have bridges, use them as candidates; otherwise fall back to all points
        if len(self._bridge_indices) > 0:
            candidate_indices = self._bridge_indices
        else:
            if verbose:
                print("    No bridges found, using all points as candidates")
            candidate_indices = None  # Will default to all points

        # If we have super connectors, use them as seeds; otherwise use None
        seed_indices = self._super_connectors.indices if len(self._super_connectors.indices) > 0 else None

        anchor_result = select_orthogonal_anchors(
            embeddings,
            k=self.n_anchors - self.include_sparse_points,
            seed_indices=seed_indices,
            candidate_indices=candidate_indices,
            use_bridges=False,  # We already handled bridge detection above
            seed=self.seed
        )

        anchor_indices = list(anchor_result.indices)

        # Step 4: Add sparse region points if requested
        if self.include_sparse_points > 0:
            if verbose:
                print(f"  Adding {self.include_sparse_points} sparse region points...")

            bucket_sizes = np.array(clf.get_bucket_sizes())
            sparse_mask = bucket_sizes < np.percentile(bucket_sizes, 25)
            sparse_indices = np.where(sparse_mask)[0]

            # Filter out points already selected
            anchor_set = set(anchor_indices)
            sparse_candidates = [i for i in sparse_indices if i not in anchor_set]

            if sparse_candidates:
                # Select spread-out sparse points
                rng = np.random.default_rng(self.seed)
                n_sparse = min(self.include_sparse_points, len(sparse_candidates))
                sparse_selected = rng.choice(sparse_candidates, n_sparse, replace=False)
                anchor_indices.extend(sparse_selected)

        self._anchor_indices = np.array(anchor_indices)
        self._anchor_embeddings = embeddings[self._anchor_indices]

        if verbose:
            print(f"    Final anchors: {len(self._anchor_indices):,}")

        # Step 5: Precompute neighborhoods for each anchor
        if verbose:
            print(f"  Precomputing {self.expansion_k}-NN neighborhoods...")

        self._neighborhoods = {}
        for anchor_idx in self._anchor_indices:
            # Find k nearest neighbors
            sims = embeddings @ embeddings[anchor_idx]
            top_k = np.argsort(sims)[-self.expansion_k:][::-1]
            self._neighborhoods[anchor_idx] = top_k

        self._fitted = True

        if verbose:
            storage_mb = self._estimate_storage_mb()
            print(f"  Index built: {storage_mb:.1f} MB")

        return self

    def query(
        self,
        query: np.ndarray,
        k: int = 10,
        n_query_anchors: Optional[int] = None,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Retrieve top-k candidates for a query embedding.

        Args:
            query: (d,) query embedding vector
            k: Number of results to return
            n_query_anchors: Override default number of anchors to probe
            return_scores: Whether to return similarity scores

        Returns:
            (indices, scores) where indices are the top-k candidate indices
            and scores are their cosine similarities (or None if return_scores=False)
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        query = np.array(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.flatten()

        n_anchors = n_query_anchors or self.n_query_anchors

        # Find nearest anchors
        anchor_sims = query @ self._anchor_embeddings.T
        top_anchor_local = np.argsort(anchor_sims)[-n_anchors:][::-1]
        top_anchors = self._anchor_indices[top_anchor_local]

        # Collect candidates from anchor neighborhoods
        candidate_set = set()
        for anchor_idx in top_anchors:
            candidate_set.update(self._neighborhoods[anchor_idx])

        candidates = np.array(list(candidate_set))

        # Score candidates
        candidate_sims = query @ self._embeddings[candidates].T

        # Return top-k
        top_k_local = np.argsort(candidate_sims)[-k:][::-1]
        top_k_indices = candidates[top_k_local]

        if return_scores:
            top_k_scores = candidate_sims[top_k_local]
            return top_k_indices, top_k_scores
        else:
            return top_k_indices, None

    def query_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_query_anchors: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch query for multiple embeddings.

        Args:
            queries: (n_queries, d) array of query embeddings
            k: Number of results per query
            n_query_anchors: Override default number of anchors to probe

        Returns:
            List of (indices, scores) tuples, one per query
        """
        results = []
        for query in queries:
            indices, scores = self.query(query, k=k, n_query_anchors=n_query_anchors)
            results.append((indices, scores))
        return results

    def get_anchors(self) -> np.ndarray:
        """Return the anchor point indices."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._anchor_indices.copy()

    def get_super_connectors(self) -> SuperConnectorResult:
        """Return the super connector analysis."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._super_connectors

    def _estimate_storage_mb(self) -> float:
        """Estimate storage size in MB."""
        if not self._fitted:
            return 0.0

        # Anchor embeddings
        anchor_bytes = self._anchor_embeddings.nbytes

        # Neighborhoods (assume int64 indices)
        neighborhood_bytes = sum(
            arr.nbytes for arr in self._neighborhoods.values()
        )

        return (anchor_bytes + neighborhood_bytes) / (1024 * 1024)

    def summary(self) -> str:
        """Return a summary of the index."""
        if not self._fitted:
            return "BridgeIndex (not fitted)"

        n_points = len(self._embeddings)
        n_anchors = len(self._anchor_indices)
        n_super = len(self._super_connectors)
        n_bridges = len(self._bridge_indices)
        storage_mb = self._estimate_storage_mb()

        avg_neighborhood = np.mean([len(v) for v in self._neighborhoods.values()])

        return (
            f"BridgeIndex: {n_points:,} points\n"
            f"  Anchors: {n_anchors:,} ({n_anchors/n_points*100:.1f}%)\n"
            f"  Super connectors: {n_super:,}\n"
            f"  Total bridges: {n_bridges:,}\n"
            f"  Avg neighborhood: {avg_neighborhood:.0f}\n"
            f"  Storage: {storage_mb:.1f} MB"
        )

    def evaluate_recall(
        self,
        n_queries: int = 100,
        k: int = 10,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate recall against brute-force search.

        Args:
            n_queries: Number of random queries to test
            k: Number of results per query
            seed: Random seed for query selection

        Returns:
            Dict with 'recall', 'avg_candidates', 'speedup' metrics
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        rng = np.random.default_rng(seed)
        query_indices = rng.choice(len(self._embeddings), n_queries, replace=False)

        total_recall = 0.0
        total_candidates = 0

        for query_idx in query_indices:
            query = self._embeddings[query_idx]

            # Brute force ground truth
            all_sims = query @ self._embeddings.T
            all_sims[query_idx] = -np.inf  # Exclude self
            gt_indices = set(np.argsort(all_sims)[-k:])

            # Index results
            idx_indices, _ = self.query(query, k=k)
            idx_set = set(idx_indices)

            # Calculate recall
            recall = len(gt_indices & idx_set) / k
            total_recall += recall

            # Count candidates (proxy via neighborhood expansion)
            anchor_sims = query @ self._anchor_embeddings.T
            top_anchors = self._anchor_indices[np.argsort(anchor_sims)[-self.n_query_anchors:]]
            candidates = set()
            for anchor in top_anchors:
                candidates.update(self._neighborhoods[anchor])
            total_candidates += len(candidates)

        avg_recall = total_recall / n_queries
        avg_candidates = total_candidates / n_queries
        speedup = len(self._embeddings) / avg_candidates if avg_candidates > 0 else 0

        return {
            'recall': avg_recall,
            'avg_candidates': avg_candidates,
            'speedup': speedup
        }


def find_super_connectors(
    embeddings: np.ndarray,
    global_num_bits: int = 12,
    facet_num_bits: int = 10,
    dense_percentile: float = 75,
    global_threshold_percentile: float = 50,
    local_threshold_percentile: float = 50,
    min_bucket_size: int = 20,
    seed: int = 42
) -> SuperConnectorResult:
    """
    Find super connectors: points with high centrality in both global and local
    bridge networks.

    Super connectors are ideal RAG anchor points because they:
    - Bridge across major semantic regions (high global centrality)
    - Connect facets within dense clusters (high local centrality)
    - Provide 10x better coverage efficiency than random anchors

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        global_num_bits: LSH bits for global bucketing
        facet_num_bits: LSH bits for facet bucketing
        dense_percentile: Percentile threshold for dense buckets
        global_threshold_percentile: Percentile for "high" global centrality
        local_threshold_percentile: Percentile for "high" local centrality
        min_bucket_size: Minimum bucket size for faceting
        seed: Random seed

    Returns:
        SuperConnectorResult with indices and centrality data
    """
    n_points, dim = embeddings.shape

    # Global DYF and bridge analysis
    global_clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
    global_clf.fit(embeddings)
    global_buckets = np.array(global_clf.get_bucket_ids())
    global_bridge = global_clf.analyze_bridges(embeddings)

    # Compute global centrality (number of buckets connected)
    global_centrality = np.zeros(n_points, dtype=np.int32)
    for i in range(len(global_bridge.bridge_indices)):
        point_idx, _, neighbors = global_bridge.get_bridge_connections(i)
        global_centrality[point_idx] = len(neighbors) + 1

    # Group by bucket for faceting
    bucket_to_indices = defaultdict(list)
    for idx, bid in enumerate(global_buckets):
        bucket_to_indices[bid].append(idx)

    bucket_counts = np.bincount(global_buckets)
    dense_threshold = np.percentile(bucket_counts[bucket_counts > 0], dense_percentile)
    dense_bucket_ids = np.where(bucket_counts > max(dense_threshold, min_bucket_size))[0]

    # Compute local centrality within dense buckets
    local_centrality = np.zeros(n_points, dtype=np.int32)

    for bid in dense_bucket_ids:
        indices = np.array(bucket_to_indices[bid])
        if len(indices) < min_bucket_size:
            continue

        bucket_emb = embeddings[indices]
        bits = 6 if len(indices) < 100 else (8 if len(indices) < 500 else facet_num_bits)

        try:
            facet_clf = DensityClassifier(embedding_dim=dim, num_bits=bits, seed=seed)
            facet_clf.fit(bucket_emb)
            facet_bridge = facet_clf.analyze_bridges(bucket_emb)

            for i in range(len(facet_bridge.bridge_indices)):
                local_idx, _, neighbors = facet_bridge.get_bridge_connections(i)
                local_centrality[indices[local_idx]] = len(neighbors) + 1
        except Exception:
            pass

    # Compute thresholds from non-zero values
    global_nonzero = global_centrality[global_centrality > 0]
    local_nonzero = local_centrality[local_centrality > 0]

    global_thresh = np.percentile(global_nonzero, global_threshold_percentile) if len(global_nonzero) > 0 else 1
    local_thresh = np.percentile(local_nonzero, local_threshold_percentile) if len(local_nonzero) > 0 else 1

    # Classify into quadrants
    quadrant = np.full(n_points, 'Regular', dtype=object)
    high_global = global_centrality > global_thresh
    high_local = local_centrality > local_thresh
    is_bridge = (global_centrality > 0) | (local_centrality > 0)

    quadrant[is_bridge & ~high_global & ~high_local] = 'Minor Bridge'
    quadrant[high_global & ~high_local] = 'Cross-Domain'
    quadrant[~high_global & high_local] = 'Domain Specialist'
    quadrant[high_global & high_local] = 'Super Connector'

    super_indices = np.where(quadrant == 'Super Connector')[0]

    return SuperConnectorResult(
        indices=super_indices,
        global_centrality=global_centrality,
        local_centrality=local_centrality,
        quadrant=quadrant,
        global_threshold=float(global_thresh),
        local_threshold=float(local_thresh)
    )


def select_orthogonal_anchors(
    embeddings: np.ndarray,
    k: int,
    seed_indices: Optional[np.ndarray] = None,
    candidate_indices: Optional[np.ndarray] = None,
    use_bridges: bool = True,
    global_num_bits: int = 12,
    seed: int = 42
) -> OrthogonalAnchorResult:
    """
    Select k maximally spread anchors using greedy farthest-point sampling.

    Achieves ~87% of full-bridge recall with ~22% of anchors by eliminating
    redundancy in anchor placement.

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        k: Number of anchors to select
        seed_indices: Initial seed points (default: super connectors)
        candidate_indices: Pool to select from (default: bridges or all points)
        use_bridges: If True and candidate_indices is None, use bridge points
        global_num_bits: LSH bits for bridge detection
        seed: Random seed

    Returns:
        OrthogonalAnchorResult with selected indices
    """
    n_points, dim = embeddings.shape

    # Get seeds (default: super connectors, or start from scratch if none)
    if seed_indices is None:
        sc_result = find_super_connectors(embeddings, global_num_bits=global_num_bits, seed=seed)
        seed_indices = sc_result.indices

    # Get candidates
    if candidate_indices is None:
        if use_bridges:
            clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
            clf.fit(embeddings)
            bridge_analysis = clf.analyze_bridges(embeddings)
            candidate_indices = np.array(bridge_analysis.bridge_indices)
            candidate_source = 'bridges'
            # Fall back to all points if no bridges found
            if len(candidate_indices) == 0:
                candidate_indices = np.arange(n_points)
                candidate_source = 'all'
        else:
            candidate_indices = np.arange(n_points)
            candidate_source = 'all'
    else:
        candidate_source = 'custom'
        # Handle empty custom candidates
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(n_points)
            candidate_source = 'all'

    # Initialize with seeds (if any)
    selected = list(seed_indices) if len(seed_indices) > 0 else []
    selected_set = set(selected)
    candidates = [c for c in candidate_indices if c not in selected_set]

    # Initialize min distances
    min_distances = np.full(n_points, np.inf)
    if selected:
        # Initialize from seeds
        for s in selected:
            dists = 1 - np.dot(embeddings, embeddings[s])
            min_distances = np.minimum(min_distances, dists)
    else:
        # No seeds - start with a random point (first candidate)
        if candidates:
            first = candidates.pop(0)
            selected.append(first)
            selected_set.add(first)
            dists = 1 - np.dot(embeddings, embeddings[first])
            min_distances = np.minimum(min_distances, dists)

    # Greedy farthest-point selection
    while len(selected) < k and candidates:
        # Find candidate farthest from all selected
        candidate_dists = min_distances[candidates]
        best_local = np.argmax(candidate_dists)
        best_idx = candidates[best_local]

        # Add to selected
        selected.append(best_idx)
        candidates.pop(best_local)

        # Update min distances
        dists = 1 - np.dot(embeddings, embeddings[best_idx])
        min_distances = np.minimum(min_distances, dists)

    return OrthogonalAnchorResult(
        indices=np.array(selected),
        seed_indices=seed_indices,
        candidate_source=candidate_source
    )


@dataclass
class FacetDiverseResult:
    """
    Result of facet-based diversification.

    Facet diversification selects one item per semantic bucket, providing
    "different perspectives on the same topic" rather than "different things".

    Attributes:
        indices: Selected item indices (one per facet)
        bucket_ids: Bucket ID for each selected item
        similarities: Query similarity for each selected item
        buckets_covered: Number of unique buckets in result
    """
    indices: np.ndarray
    bucket_ids: np.ndarray
    similarities: np.ndarray
    buckets_covered: int

    def __len__(self) -> int:
        return len(self.indices)


def diversify_by_facet(
    query: np.ndarray,
    candidate_indices: np.ndarray,
    embeddings: np.ndarray,
    bucket_ids: np.ndarray,
    k: int = 10,
    similarity_weight: float = 0.0,
) -> FacetDiverseResult:
    """
    Select k items from k different semantic facets (buckets).

    Unlike MMR which maximizes geometric diversity ("different things"),
    facet diversification maximizes semantic coverage ("different perspectives").
    Each result comes from a different DYF bucket, ensuring coverage of distinct
    semantic regions.

    This is preferred 2:1 over MMR for queries with high topical redundancy
    (same concept appearing in multiple results).

    Args:
        query: Query embedding vector (d,)
        candidate_indices: Indices of candidate items to diversify
        embeddings: Full embedding matrix (n, d)
        bucket_ids: DYF bucket IDs for all items (from classifier.get_bucket_ids())
        k: Number of diverse results to return
        similarity_weight: Weight for similarity vs bucket order (0.0 = pure bucket
            diversity, 1.0 = similarity-weighted selection within buckets)

    Returns:
        FacetDiverseResult with selected indices and metadata

    Example:
        >>> from dyf import DensityClassifier, diversify_by_facet
        >>>
        >>> # Fit classifier
        >>> clf = DensityClassifier(embedding_dim=dim)
        >>> clf.fit(embeddings)
        >>> bucket_ids = clf.get_bucket_ids()
        >>>
        >>> # Get initial candidates (e.g., top-100 by similarity)
        >>> sims = query @ embeddings.T
        >>> candidates = np.argsort(sims)[-100:][::-1]
        >>>
        >>> # Diversify by facet
        >>> result = diversify_by_facet(query, candidates, embeddings, bucket_ids, k=10)
        >>> print(f"Selected {len(result)} items from {result.buckets_covered} buckets")
    """
    query = np.array(query, dtype=np.float32).flatten()
    if np.linalg.norm(query) > 0:
        query = query / np.linalg.norm(query)

    candidate_indices = np.array(candidate_indices)
    if len(candidate_indices) == 0:
        return FacetDiverseResult(
            indices=np.array([], dtype=np.int64),
            bucket_ids=np.array([], dtype=np.int64),
            similarities=np.array([], dtype=np.float32),
            buckets_covered=0
        )

    # Compute similarities for candidates
    candidate_embeddings = embeddings[candidate_indices]
    similarities = candidate_embeddings @ query

    # Sort candidates by similarity (descending)
    sorted_order = np.argsort(similarities)[::-1]
    sorted_candidates = candidate_indices[sorted_order]
    sorted_sims = similarities[sorted_order]

    # Select one per bucket
    selected_indices = []
    selected_bucket_ids = []
    selected_sims = []
    seen_buckets = set()

    for i, (idx, sim) in enumerate(zip(sorted_candidates, sorted_sims)):
        bid = bucket_ids[idx]
        if bid not in seen_buckets:
            selected_indices.append(idx)
            selected_bucket_ids.append(bid)
            selected_sims.append(sim)
            seen_buckets.add(bid)

            if len(selected_indices) >= k:
                break

    return FacetDiverseResult(
        indices=np.array(selected_indices, dtype=np.int64),
        bucket_ids=np.array(selected_bucket_ids, dtype=np.int64),
        similarities=np.array(selected_sims, dtype=np.float32),
        buckets_covered=len(seen_buckets)
    )


def get_kmeans_init(
    embeddings: np.ndarray,
    nlist: int,
    use_stable_bridges: bool = True,
    num_stability_seeds: int = 5,
    stability_threshold: int = 3,
    global_num_bits: int = 12,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """
    Get bridge-seeded initial centroids for k-means/IVF index construction.

    Using orthogonal bridge points as k-means initialization provides:
    - +1% recall improvement over standard k-means++
    - 17-24% fewer bucket probes needed for same recall
    - Better bucket coherence (neighbors land in same bucket more often)

    The returned centroids can be used with sklearn.cluster.KMeans or FAISS IVF.

    Args:
        embeddings: Normalized embeddings (n, d)
        nlist: Number of clusters/centroids (IVF nlist parameter)
        use_stable_bridges: If True, prefer bridges stable across multiple seeds
        num_stability_seeds: Number of seeds for stability testing
        stability_threshold: Minimum seeds a bridge must appear in (if use_stable_bridges)
        global_num_bits: LSH bits for bridge detection
        seed: Random seed
        verbose: Print progress

    Returns:
        Initial centroids array (nlist, d) suitable for k-means init parameter

    Example:
        >>> from dyf import get_kmeans_init
        >>> from sklearn.cluster import KMeans
        >>>
        >>> # Get bridge-seeded initialization
        >>> init = get_kmeans_init(embeddings, nlist=1000)
        >>>
        >>> # Use with sklearn
        >>> kmeans = KMeans(n_clusters=1000, init=init, n_init=1)
        >>> kmeans.fit(embeddings)
        >>>
        >>> # Use with FAISS
        >>> import faiss
        >>> quantizer = faiss.IndexFlatIP(dim)
        >>> quantizer.add(kmeans.cluster_centers_)

    Note:
        For high-recall requirements (99%+), consider using
        `candidate_indices=np.arange(len(embeddings))` in select_orthogonal_anchors
        instead, as all-points expansion provides better long-tail coverage.
    """
    n_points, dim = embeddings.shape

    # Normalize if needed
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    if verbose:
        print(f"Computing bridge-seeded k-means initialization (nlist={nlist})...")

    # Find candidate bridges
    if use_stable_bridges and num_stability_seeds > 1:
        if verbose:
            print(f"  Finding stable bridges ({num_stability_seeds} seeds)...")

        # Count bridge appearances across seeds
        bridge_counts = np.zeros(n_points, dtype=np.int32)

        for seed_idx in range(num_stability_seeds):
            seed_offset = seed_idx * 1000
            clf = DensityClassifier(
                embedding_dim=dim,
                num_bits=global_num_bits,
                seed=seed + seed_offset
            )
            clf.fit(embeddings)
            bridge_analysis = clf.analyze_bridges(embeddings)
            for bridge_idx in bridge_analysis.bridge_indices:
                bridge_counts[bridge_idx] += 1

        # Select stable bridges
        stable_mask = bridge_counts >= stability_threshold
        stable_bridges = np.where(stable_mask)[0]

        if verbose:
            total_bridges = (bridge_counts > 0).sum()
            print(f"    Total bridges: {total_bridges}, stable: {len(stable_bridges)}")

        if len(stable_bridges) >= nlist:
            candidate_indices = stable_bridges
        else:
            # Fall back to all bridges
            candidate_indices = np.where(bridge_counts > 0)[0]
    else:
        # Use single-seed bridges
        clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
        clf.fit(embeddings)
        bridge_analysis = clf.analyze_bridges(embeddings)
        candidate_indices = np.array(bridge_analysis.bridge_indices)

        if verbose:
            print(f"  Found {len(candidate_indices)} bridges")

    # Fall back to all points if not enough candidates
    if len(candidate_indices) < nlist:
        if verbose:
            print(f"  Not enough candidates ({len(candidate_indices)}), using all points")
        candidate_indices = np.arange(n_points)

    # Select orthogonal anchors
    if verbose:
        print(f"  Selecting {nlist} orthogonal anchors...")

    anchor_result = select_orthogonal_anchors(
        embeddings,
        k=nlist,
        candidate_indices=candidate_indices,
        use_bridges=False,  # Already handled above
        seed=seed
    )

    # Return anchor embeddings as initial centroids
    init_centroids = embeddings[anchor_result.indices].copy()

    # Pad if we didn't get enough anchors
    if len(init_centroids) < nlist:
        if verbose:
            print(f"  Padding from {len(init_centroids)} to {nlist} centroids")
        rng = np.random.default_rng(seed)
        remaining = nlist - len(init_centroids)
        existing_set = set(anchor_result.indices)
        available = [i for i in range(n_points) if i not in existing_set]
        extra = rng.choice(available, min(remaining, len(available)), replace=False)
        init_centroids = np.vstack([init_centroids, embeddings[extra]])

    if verbose:
        print(f"  Done. Returned {len(init_centroids)} initial centroids")

    return init_centroids.astype(np.float32)


# ============================================================
# DAG Mining
# ============================================================


@dataclass
class DAGChain:
    """
    A single DAG chain representing a hierarchical path.

    Chains flow from general concepts (high neighbor diversity) to
    specific concepts (low neighbor diversity).

    Attributes:
        indices: Point indices in chain order (general → specific)
        coherence: Average pairwise similarity within chain (higher = tighter)
        diversity_range: Difference between first and last diversity scores
    """
    indices: np.ndarray
    coherence: float
    diversity_range: float

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class DAGMiningResult:
    """
    Result of DAG mining on an embedding space.

    DAG mining finds hierarchical structures by using neighbor diversity
    as a generality signal. Points with diverse neighbors (connecting
    many topics) are considered "general", while points with coherent
    neighbors (tight clusters) are "specific".

    Attributes:
        chains: List of extracted DAG chains
        diversity: Neighbor diversity score for each point
        parent_child_edges: List of (parent, child, similarity, gap) tuples
        n_components: Number of connected components in the DAG
    """
    chains: List[DAGChain]
    diversity: np.ndarray
    parent_child_edges: List[Tuple[int, int, float, float]]
    n_components: int

    def __len__(self) -> int:
        return len(self.chains)

    def summary(self) -> str:
        """Return summary statistics."""
        if not self.chains:
            return "DAGMiningResult: No chains found"

        lengths = [len(c) for c in self.chains]
        coherences = [c.coherence for c in self.chains]

        return (
            f"DAGMiningResult:\n"
            f"  Chains: {len(self.chains)}\n"
            f"  Length: min={min(lengths)}, max={max(lengths)}, "
            f"avg={np.mean(lengths):.1f}\n"
            f"  Coherence: min={min(coherences):.2f}, max={max(coherences):.2f}, "
            f"avg={np.mean(coherences):.2f}\n"
            f"  Components: {self.n_components}\n"
            f"  Parent-child edges: {len(self.parent_child_edges)}"
        )

    def get_chains_by_length(self, min_length: int = 3) -> List[DAGChain]:
        """Return chains with at least min_length nodes."""
        return [c for c in self.chains if len(c) >= min_length]

    def get_chains_by_coherence(self, min_coherence: float = 0.65) -> List[DAGChain]:
        """Return chains with at least min_coherence."""
        return [c for c in self.chains if c.coherence >= min_coherence]


def compute_neighbor_diversity(
    embeddings: np.ndarray,
    k: int = 15,
    neighbors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute neighbor diversity for each point.

    Diversity measures how dissimilar a point's neighbors are to each other.
    High diversity indicates a "general" concept that connects multiple topics.
    Low diversity indicates a "specific" concept within a tight cluster.

    Args:
        embeddings: Normalized embeddings (n, d)
        k: Number of neighbors to consider
        neighbors: Pre-computed k-NN indices (n, k+1). If None, computed internally.

    Returns:
        Diversity scores for each point (n,). Higher = more general.

    Example:
        >>> diversity = compute_neighbor_diversity(embeddings, k=15)
        >>> general_idx = np.argsort(diversity)[-10:]  # Most general concepts
        >>> specific_idx = np.argsort(diversity)[:10]  # Most specific concepts
    """
    n_points, dim = embeddings.shape

    # Compute k-NN if not provided
    if neighbors is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(embeddings)
        _, neighbors = nn.kneighbors(embeddings)

    diversity = np.zeros(n_points)

    for i in range(n_points):
        # Get neighbor embeddings (exclude self)
        neighb_idx = neighbors[i, 1:k + 1]
        neighb_emb = embeddings[neighb_idx]

        # Compute pairwise similarity among neighbors
        pairwise = neighb_emb @ neighb_emb.T

        # Get off-diagonal (neighbor-to-neighbor similarity)
        mask = ~np.eye(len(neighb_idx), dtype=bool)
        neighbor_coherence = pairwise[mask].mean()

        # Diversity = 1 - coherence (higher diversity = more general)
        diversity[i] = 1 - neighbor_coherence

    return diversity


@dataclass
class HubScoreResult:
    """
    Result of hub score computation.

    Hub nodes are structural centers that hold the embedding space together.
    They are characterized by:
    - High eigenvector centrality (in tightly woven neighborhoods)
    - Low distance to global centroid (central position in space)

    Unlike PageRank, which rewards being pointed to by many nodes (both categories
    AND popular instances qualify), eigenvector centrality rewards being in a
    tightly interconnected cluster. Abstract concepts cluster more tightly than
    popular instances.

    Attributes:
        scores: Hub score for each point (higher = more structural/general)
        eigenvector: Eigenvector centrality for each point
        distance_to_center: Distance to global centroid for each point
        eigenvector_z: Z-scored eigenvector centrality
        distance_z: Z-scored distance to center
    """
    scores: np.ndarray
    eigenvector: np.ndarray
    distance_to_center: np.ndarray
    eigenvector_z: np.ndarray
    distance_z: np.ndarray

    def get_hub_nodes(self, threshold: float = 0.0) -> np.ndarray:
        """Return indices of nodes with hub score above threshold."""
        return np.where(self.scores > threshold)[0]

    def get_top_hubs(self, k: int = 100) -> np.ndarray:
        """Return indices of top k hub nodes."""
        return np.argsort(self.scores)[-k:][::-1]


def compute_hub_score(
    embeddings: np.ndarray,
    k: int = 30,
    n_iterations: int = 30,
    neighbors: Optional[np.ndarray] = None,
    similarities: Optional[np.ndarray] = None,
) -> HubScoreResult:
    """
    Compute hub score for each point in the embedding space.

    Hub score identifies structural hub nodes that hold the embedding space
    together, as opposed to popular instances that fill it. The score combines:
    - Eigenvector centrality: rewards being in a tightly woven neighborhood
    - Distance to global centroid: rewards central position in the space

    General/structural concepts score high on both signals; specific instances
    score low on both.

    Args:
        embeddings: Normalized embeddings (n, d)
        k: Number of neighbors for k-NN graph
        n_iterations: Iterations for eigenvector power method
        neighbors: Pre-computed k-NN indices (n, k+1). If None, computed internally.
        similarities: Pre-computed similarities (n, k+1). If None, computed internally.

    Returns:
        HubScoreResult with scores and component signals.

    Example:
        >>> result = compute_hub_score(embeddings)
        >>> hub_nodes = result.get_hub_nodes(threshold=1.0)
        >>> print(f"Found {len(hub_nodes)} hub nodes")
        >>>
        >>> # Use for taxonomy direction
        >>> if result.scores[idx_animal] > result.scores[idx_dog]:
        ...     print("Animal is more general than Dog")
    """
    n_points, dim = embeddings.shape

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Compute k-NN if not provided
    if neighbors is None or similarities is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(embeddings)
        distances, neighbors = nn.kneighbors(embeddings)
        similarities = 1 - distances

    # 1. Eigenvector centrality (no damping - key difference from PageRank)
    eigen = np.ones(n_points)
    for _ in range(n_iterations):
        new_eigen = np.zeros(n_points)
        for i in range(n_points):
            for j in range(1, k + 1):
                if j < neighbors.shape[1]:
                    new_eigen[i] += eigen[neighbors[i, j]] * similarities[i, j]
        norm = np.linalg.norm(new_eigen)
        if norm > 0:
            eigen = new_eigen / norm
        else:
            break

    # 2. Distance to global centroid
    global_centroid = embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(global_centroid)
    if centroid_norm > 0:
        global_centroid = global_centroid / centroid_norm
    distance_to_center = 1 - np.dot(embeddings, global_centroid)

    # 3. Z-score normalize
    eigen_z = (eigen - eigen.mean()) / (eigen.std() + 1e-10)
    dist_z = (distance_to_center - distance_to_center.mean()) / (distance_to_center.std() + 1e-10)

    # 4. Combine: high eigenvector, close to center (low distance)
    hub_score = eigen_z - dist_z

    return HubScoreResult(
        scores=hub_score,
        eigenvector=eigen,
        distance_to_center=distance_to_center,
        eigenvector_z=eigen_z,
        distance_z=dist_z,
    )


def mine_dag_chains(
    embeddings: np.ndarray,
    k_neighbors: int = 30,
    similarity_threshold: float = 0.55,
    diversity_gap_threshold: float = 0.02,
    min_chain_length: int = 3,
    verbose: bool = False,
) -> DAGMiningResult:
    """
    Mine DAG (directed acyclic graph) structures from embedding space.

    Finds hierarchical chains by using neighbor diversity as a generality
    signal. Points with diverse neighbors connect many topics (general),
    while points with coherent neighbors form tight clusters (specific).

    Chains flow from general → specific, following the diversity gradient.

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        k_neighbors: Number of neighbors for k-NN graph and diversity computation.
        similarity_threshold: Minimum similarity for parent-child edges.
        diversity_gap_threshold: Minimum diversity difference for directed edge.
        min_chain_length: Minimum chain length to return.
        verbose: Print progress information.

    Returns:
        DAGMiningResult with chains, diversity scores, and edge information.

    Example:
        >>> from dyf import mine_dag_chains
        >>>
        >>> result = mine_dag_chains(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Get clean hierarchies
        >>> clean = result.get_chains_by_coherence(min_coherence=0.65)
        >>> for chain in clean[:10]:
        ...     print(f"[len={len(chain)}] {chain.indices}")

    Notes:
        - Low diversity points (decades, months) have highly coherent neighbors
        - High diversity points sit at intersections of multiple topics
        - Chains often converge to common "sinks" (abstract concepts)
        - ~100% of extracted chains follow monotonic diversity gradient
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize embeddings
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    n_points, dim = embeddings.shape

    if verbose:
        print(f"Mining DAG chains: {n_points:,} points, dim={dim}")

    # Build k-NN graph
    if verbose:
        print("  Building k-NN graph...")
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    # Compute neighbor diversity
    if verbose:
        print("  Computing neighbor diversity...")
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    if verbose:
        print(f"    Diversity range: {diversity.min():.3f} to {diversity.max():.3f}")

    # Find parent-child pairs based on diversity gradient
    if verbose:
        print("  Finding parent-child pairs...")

    parent_child_edges = []

    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i, 1:16]):
            sim = similarities[i, j_idx + 1]
            if sim < similarity_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            # Parent has higher diversity (more general)
            if gap > diversity_gap_threshold:
                parent_child_edges.append((i, j, float(sim), float(gap)))
            elif gap < -diversity_gap_threshold:
                parent_child_edges.append((j, i, float(sim), float(-gap)))

    if verbose:
        print(f"    Found {len(parent_child_edges)} parent-child pairs")

    # Build adjacency lists
    children = defaultdict(list)
    parents = defaultdict(list)

    for parent, child, sim, gap in parent_child_edges:
        children[parent].append((child, sim, gap))
        parents[child].append((parent, sim, gap))

    # Find connected components using union-find
    parent_map = list(range(n_points))

    def find(x):
        if parent_map[x] != x:
            parent_map[x] = find(parent_map[x])
        return parent_map[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent_map[px] = py

    for parent, child, _, _ in parent_child_edges:
        union(parent, child)

    components = defaultdict(set)
    for i in range(n_points):
        if i in children or i in parents:
            components[find(i)].add(i)

    components = {k: v for k, v in components.items() if len(v) >= min_chain_length}
    n_components = len(components)

    if verbose:
        print(f"    Found {n_components} non-trivial components")

    # Extract chains from components
    if verbose:
        print("  Extracting chains...")

    def extract_chains_from_component(component_nodes):
        """Extract maximal chains from a connected component."""
        # Find roots (nodes with no parents in this component)
        roots = [
            n for n in component_nodes
            if n not in parents or all(p not in component_nodes for p, _, _ in parents[n])
        ]

        if not roots:
            # No clear root - pick highest diversity node
            roots = [max(component_nodes, key=lambda x: diversity[x])]

        all_chains = []
        for root in roots:
            chain = [root]
            current = root
            visited = {root}

            while True:
                if current not in children:
                    break
                valid = [
                    (c, s, g) for c, s, g in children[current]
                    if c in component_nodes and c not in visited
                ]
                if not valid:
                    break
                # Pick by highest similarity
                valid.sort(key=lambda x: x[1], reverse=True)
                next_node = valid[0][0]
                chain.append(next_node)
                visited.add(next_node)
                current = next_node

            if len(chain) >= min_chain_length:
                all_chains.append(chain)

        return all_chains

    chains = []
    for comp_nodes in components.values():
        comp_chains = extract_chains_from_component(comp_nodes)
        for chain in comp_chains:
            # Compute chain metrics
            chain_emb = embeddings[chain]
            coherence = float((chain_emb @ chain_emb.T).mean())
            div_range = float(diversity[chain[0]] - diversity[chain[-1]])

            chains.append(DAGChain(
                indices=np.array(chain),
                coherence=coherence,
                diversity_range=div_range
            ))

    # Sort by quality (coherence * length)
    chains.sort(key=lambda x: x.coherence * len(x), reverse=True)

    if verbose:
        print(f"    Extracted {len(chains)} chains (length >= {min_chain_length})")
        if chains:
            clean = [c for c in chains if c.coherence >= 0.65 and len(c) >= 4]
            print(f"    Clean hierarchies (coh >= 0.65, len >= 4): {len(clean)}")

    return DAGMiningResult(
        chains=chains,
        diversity=diversity,
        parent_child_edges=parent_child_edges,
        n_components=n_components
    )


# ============================================================
# DAG Taxonomy
# ============================================================


@dataclass
class DAGTaxonomy:
    """
    A navigable DAG taxonomy extracted from embedding space.

    Provides lattice operations for exploring hierarchical relationships:
    - Find ancestors/descendants of a concept
    - Find common ancestors between concepts
    - Find paths through the taxonomy
    - Identify convergence points (abstract attractors)

    The taxonomy is a true lattice (not a tree) - nodes can have multiple
    parents, enabling multiple inheritance and diamond patterns.

    Attributes:
        n_nodes: Total number of nodes in the taxonomy
        diversity: Neighbor diversity score for each node
        children: Dict mapping node → list of (child, similarity, gap)
        parents: Dict mapping node → list of (parent, similarity, gap)
        roots: Nodes with no parents (most general concepts)
        leaves: Nodes with no children (most specific concepts)

    Example:
        >>> from dyf import build_dag_taxonomy
        >>>
        >>> taxonomy = build_dag_taxonomy(embeddings, verbose=True)
        >>> print(taxonomy.summary())
        >>>
        >>> # Find ancestors of a concept
        >>> ancestors = taxonomy.get_ancestors(node_idx, max_depth=5)
        >>>
        >>> # Find common ancestors between two concepts
        >>> common = taxonomy.get_common_ancestors(idx_a, idx_b)
        >>>
        >>> # Get convergence points (highly connected abstract concepts)
        >>> hubs = taxonomy.get_convergence_points(min_parents=10)
    """
    n_nodes: int
    diversity: np.ndarray
    children: Dict[int, List[Tuple[int, float, float]]]
    parents: Dict[int, List[Tuple[int, float, float]]]
    roots: List[int]
    leaves: List[int]

    def __len__(self) -> int:
        """Number of nodes with edges."""
        return len(set(self.children.keys()) | set(self.parents.keys()))

    def summary(self) -> str:
        """Return summary statistics."""
        n_edges = sum(len(v) for v in self.children.values())
        n_multi_parent = sum(1 for v in self.parents.values() if len(v) > 1)
        max_parents = max((len(v) for v in self.parents.values()), default=0)
        max_children = max((len(v) for v in self.children.values()), default=0)

        return (
            f"DAGTaxonomy:\n"
            f"  Nodes with edges: {len(self)}\n"
            f"  Total edges: {n_edges}\n"
            f"  Roots (no parents): {len(self.roots)}\n"
            f"  Leaves (no children): {len(self.leaves)}\n"
            f"  Multi-parent nodes: {n_multi_parent} ({100*n_multi_parent/max(len(self),1):.1f}%)\n"
            f"  Max parents: {max_parents}\n"
            f"  Max children: {max_children}"
        )

    def get_children(self, node: int) -> List[int]:
        """Get direct children of a node."""
        return [c for c, _, _ in self.children.get(node, [])]

    def get_parents(self, node: int) -> List[int]:
        """Get direct parents of a node."""
        return [p for p, _, _ in self.parents.get(node, [])]

    def get_ancestors(self, node: int, max_depth: int = 10) -> set:
        """
        Get all ancestors of a node (transitive closure upward).

        Args:
            node: Starting node index
            max_depth: Maximum depth to traverse

        Returns:
            Set of all ancestor node indices
        """
        ancestors = set()
        frontier = {node}
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = set()
            for n in frontier:
                for parent, _, _ in self.parents.get(n, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier
            depth += 1

        return ancestors

    def get_descendants(self, node: int, max_depth: int = 10) -> set:
        """
        Get all descendants of a node (transitive closure downward).

        Args:
            node: Starting node index
            max_depth: Maximum depth to traverse

        Returns:
            Set of all descendant node indices
        """
        descendants = set()
        frontier = {node}
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = set()
            for n in frontier:
                for child, _, _ in self.children.get(n, []):
                    if child not in descendants:
                        descendants.add(child)
                        next_frontier.add(child)
            frontier = next_frontier
            depth += 1

        return descendants

    def get_common_ancestors(self, node_a: int, node_b: int, max_depth: int = 10) -> set:
        """
        Find common ancestors of two nodes.

        In a lattice (vs tree), there can be multiple common ancestors,
        not just a single LCA.

        Args:
            node_a: First node index
            node_b: Second node index
            max_depth: Maximum depth to search

        Returns:
            Set of common ancestor indices
        """
        ancestors_a = self.get_ancestors(node_a, max_depth)
        ancestors_b = self.get_ancestors(node_b, max_depth)
        return ancestors_a & ancestors_b

    def get_lowest_common_ancestors(self, node_a: int, node_b: int, max_depth: int = 10) -> List[int]:
        """
        Find lowest common ancestors (LCAs) of two nodes.

        Returns ancestors that are common to both nodes but have no
        descendants that are also common ancestors.

        Args:
            node_a: First node index
            node_b: Second node index
            max_depth: Maximum depth to search

        Returns:
            List of LCA indices (may be multiple in a lattice)
        """
        common = self.get_common_ancestors(node_a, node_b, max_depth)
        if not common:
            return []

        # Filter to those with no common descendants
        lcas = []
        for ancestor in common:
            descendants = self.get_descendants(ancestor, max_depth)
            # If none of its descendants are also common ancestors, it's an LCA
            if not (descendants & common):
                lcas.append(ancestor)

        return lcas

    def get_convergence_points(self, min_parents: int = 5) -> List[Tuple[int, int]]:
        """
        Find convergence points (nodes with many incoming edges).

        These are abstract concepts where many paths converge.

        Args:
            min_parents: Minimum number of parents to qualify

        Returns:
            List of (node_idx, parent_count) tuples, sorted by count descending
        """
        convergence = [
            (node, len(parents))
            for node, parents in self.parents.items()
            if len(parents) >= min_parents
        ]
        convergence.sort(key=lambda x: x[1], reverse=True)
        return convergence

    def get_divergence_points(self, min_children: int = 5) -> List[Tuple[int, int]]:
        """
        Find divergence points (nodes with many outgoing edges).

        These are branching hubs where paths diverge.

        Args:
            min_children: Minimum number of children to qualify

        Returns:
            List of (node_idx, child_count) tuples, sorted by count descending
        """
        divergence = [
            (node, len(children))
            for node, children in self.children.items()
            if len(children) >= min_children
        ]
        divergence.sort(key=lambda x: x[1], reverse=True)
        return divergence

    def get_diamond_patterns(self, limit: int = 100) -> List[Tuple[int, int, int, int]]:
        """
        Find diamond patterns (A → B, A → C, B → D, C → D).

        Diamonds indicate multiple inheritance / non-tree structure.

        Args:
            limit: Maximum number of diamonds to return

        Returns:
            List of (top, left, right, bottom) tuples
        """
        diamonds = []

        for d in self.parents:  # potential bottom of diamond
            d_parents = self.get_parents(d)
            if len(d_parents) < 2:
                continue

            # Check if any two parents share a common parent (top of diamond)
            for i, p1 in enumerate(d_parents):
                for p2 in d_parents[i + 1:]:
                    common_grandparents = set(self.get_parents(p1)) & set(self.get_parents(p2))
                    for a in common_grandparents:
                        diamonds.append((a, p1, p2, d))
                        if len(diamonds) >= limit:
                            return diamonds

        return diamonds

    def get_path(self, start: int, end: int, max_depth: int = 10) -> Optional[List[int]]:
        """
        Find a path from start to end (following parent→child direction).

        Args:
            start: Starting node (more general)
            end: Ending node (more specific)
            max_depth: Maximum path length

        Returns:
            List of node indices from start to end, or None if no path
        """
        if start == end:
            return [start]

        # BFS from start
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            for child, _, _ in self.children.get(current, []):
                if child == end:
                    return path + [child]
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        return None

    def get_all_paths(self, start: int, end: int, max_depth: int = 10) -> List[List[int]]:
        """
        Find all paths from start to end.

        In a lattice, there can be multiple paths between nodes.

        Args:
            start: Starting node (more general)
            end: Ending node (more specific)
            max_depth: Maximum path length

        Returns:
            List of paths, where each path is a list of node indices
        """
        if start == end:
            return [[start]]

        paths = []

        def dfs(current: int, path: List[int]):
            if len(path) > max_depth:
                return
            if current == end:
                paths.append(path.copy())
                return

            for child, _, _ in self.children.get(current, []):
                if child not in path:  # avoid cycles (shouldn't happen in DAG)
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(start, [start])
        return paths

    def get_node_depth(self, node: int) -> int:
        """
        Get the depth of a node (distance from nearest root).

        Args:
            node: Node index

        Returns:
            Minimum distance to any root, or -1 if not connected
        """
        if node in self.roots:
            return 0

        depth = 0
        frontier = {node}
        visited = {node}

        while frontier:
            depth += 1
            if depth > self.n_nodes:  # safety limit
                return -1

            next_frontier = set()
            for n in frontier:
                for parent, _, _ in self.parents.get(n, []):
                    if parent in self.roots:
                        return depth
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier

        return -1  # not connected to any root


def build_dag_taxonomy(
    embeddings: np.ndarray,
    k_neighbors: int = 30,
    similarity_threshold: float = 0.55,
    diversity_gap_threshold: float = 0.02,
    verbose: bool = False,
) -> DAGTaxonomy:
    """
    Build a navigable DAG taxonomy from embedding space.

    Creates a lattice structure where:
    - Nodes with high neighbor diversity are "general" (parents)
    - Nodes with low neighbor diversity are "specific" (children)
    - Multiple parents are allowed (non-tree structure)

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        k_neighbors: Number of neighbors for k-NN graph.
        similarity_threshold: Minimum similarity for parent-child edges.
        diversity_gap_threshold: Minimum diversity difference for directed edge.
        verbose: Print progress information.

    Returns:
        DAGTaxonomy with navigation methods.

    Example:
        >>> from dyf import build_dag_taxonomy
        >>>
        >>> taxonomy = build_dag_taxonomy(embeddings)
        >>>
        >>> # Navigate the taxonomy
        >>> ancestors = taxonomy.get_ancestors(query_idx)
        >>> common = taxonomy.get_common_ancestors(idx_a, idx_b)
        >>> hubs = taxonomy.get_convergence_points(min_parents=10)
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize embeddings
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    n_points, dim = embeddings.shape

    if verbose:
        print(f"Building DAG taxonomy: {n_points:,} points, dim={dim}")

    # Build k-NN graph
    if verbose:
        print("  Building k-NN graph...")
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    # Compute neighbor diversity
    if verbose:
        print("  Computing neighbor diversity...")
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    # Find parent-child pairs
    if verbose:
        print("  Finding parent-child relationships...")

    children: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    parents: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i, 1:16]):
            sim = similarities[i, j_idx + 1]
            if sim < similarity_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            # Parent has higher diversity (more general)
            if gap > diversity_gap_threshold:
                children[i].append((j, float(sim), float(gap)))
                parents[j].append((i, float(sim), float(gap)))
            elif gap < -diversity_gap_threshold:
                children[j].append((i, float(sim), float(-gap)))
                parents[i].append((j, float(sim), float(-gap)))

    # Convert to regular dicts
    children = dict(children)
    parents = dict(parents)

    # Find roots (no parents) and leaves (no children)
    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    if verbose:
        n_edges = sum(len(v) for v in children.values())
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        print(f"    Nodes: {len(all_nodes)}, Edges: {n_edges}")
        print(f"    Roots: {len(roots)}, Leaves: {len(leaves)}")
        print(f"    Multi-parent nodes: {n_multi} ({100*n_multi/max(len(all_nodes),1):.1f}%)")

    return DAGTaxonomy(
        n_nodes=n_points,
        diversity=diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )


@dataclass
class UnifiedOntologyResult:
    """
    Result of building a unified ontology from embedding space.

    The unified ontology combines:
    1. Main ontology (high similarity threshold) for dense regions
    2. Outlier ontology (lower threshold) for sparse regions
    3. Bridge edges connecting the two

    Attributes:
        ontology: The unified DAGTaxonomy
        main_nodes: Indices of nodes from main ontology
        outlier_nodes: Indices of nodes from outlier ontology
        excluded_nodes: Indices of double outliers (excluded)
        bridge_edges: Number of edges connecting main and outlier components
        main_threshold: Similarity threshold used for main ontology
        outlier_threshold: Similarity threshold used for outlier ontology
    """
    ontology: DAGTaxonomy
    main_nodes: np.ndarray
    outlier_nodes: np.ndarray
    excluded_nodes: np.ndarray
    bridge_edges: int
    main_threshold: float
    outlier_threshold: float

    def __len__(self) -> int:
        return len(self.ontology)

    def summary(self) -> str:
        """Return summary statistics."""
        total = self.ontology.n_nodes
        covered = len(self.main_nodes) + len(self.outlier_nodes)
        n_edges = sum(len(v) for v in self.ontology.children.values())

        return (
            f"UnifiedOntologyResult:\n"
            f"  Coverage: {covered}/{total} ({100*covered/total:.1f}%)\n"
            f"  Main nodes (sim≥{self.main_threshold}): {len(self.main_nodes)}\n"
            f"  Outlier nodes (sim≥{self.outlier_threshold}): {len(self.outlier_nodes)}\n"
            f"  Bridge edges: {self.bridge_edges}\n"
            f"  Excluded (double outliers): {len(self.excluded_nodes)}\n"
            f"  Total edges: {n_edges}\n"
            f"  Multi-parent: {sum(1 for v in self.ontology.parents.values() if len(v) > 1)}"
        )


def build_unified_ontology(
    embeddings: np.ndarray,
    main_similarity_threshold: float = 0.55,
    outlier_similarity_threshold: float = 0.45,
    diversity_gap_threshold: float = 0.02,
    outlier_diversity_gap: float = 0.015,
    k_neighbors: int = 30,
    verbose: bool = False,
) -> UnifiedOntologyResult:
    """
    Build a unified ontology that covers both dense and sparse embedding regions.

    Uses a two-tier approach:
    1. Build main ontology with high similarity threshold (dense regions)
    2. Build outlier ontology with lower threshold (sparse regions)
    3. Add bridge edges to connect them
    4. Exclude "double outliers" that don't fit either

    This achieves ~96% coverage vs ~89% for single-threshold approach.

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        main_similarity_threshold: Similarity threshold for main ontology (default 0.55).
        outlier_similarity_threshold: Similarity threshold for outlier ontology (default 0.45).
        diversity_gap_threshold: Diversity gap for main ontology edges (default 0.02).
        outlier_diversity_gap: Diversity gap for outlier ontology edges (default 0.015).
        k_neighbors: Number of neighbors for k-NN graph.
        verbose: Print progress information.

    Returns:
        UnifiedOntologyResult with combined ontology and metadata.

    Example:
        >>> from dyf import build_unified_ontology
        >>>
        >>> result = build_unified_ontology(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Access the unified ontology
        >>> ontology = result.ontology
        >>> ancestors = ontology.get_ancestors(node_idx)
        >>>
        >>> # Check which tier a node came from
        >>> if node_idx in result.main_nodes:
        ...     print("From main ontology")
        >>> elif node_idx in result.outlier_nodes:
        ...     print("From outlier ontology")
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize embeddings
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    n_points, dim = embeddings.shape

    if verbose:
        print(f"Building unified ontology: {n_points:,} points, dim={dim}")

    # Step 1: Build main ontology
    if verbose:
        print(f"\n1. Building main ontology (sim≥{main_similarity_threshold})...")

    main_ont = build_dag_taxonomy(
        embeddings,
        k_neighbors=k_neighbors,
        similarity_threshold=main_similarity_threshold,
        diversity_gap_threshold=diversity_gap_threshold,
        verbose=False,
    )

    main_connected = set(main_ont.children.keys()) | set(main_ont.parents.keys())
    main_outlier_indices = list(set(range(n_points)) - main_connected)

    if verbose:
        print(f"   Main: {len(main_connected)} nodes, {sum(len(v) for v in main_ont.children.values())} edges")
        print(f"   Outliers: {len(main_outlier_indices)} nodes")

    # Step 2: Build outlier ontology
    if verbose:
        print(f"\n2. Building outlier ontology (sim≥{outlier_similarity_threshold})...")

    if len(main_outlier_indices) > 0:
        outlier_embeddings = embeddings[main_outlier_indices]
        outlier_ont = build_dag_taxonomy(
            outlier_embeddings,
            k_neighbors=min(k_neighbors, len(main_outlier_indices) - 1),
            similarity_threshold=outlier_similarity_threshold,
            diversity_gap_threshold=outlier_diversity_gap,
            verbose=False,
        )

        outlier_connected = set(outlier_ont.children.keys()) | set(outlier_ont.parents.keys())
        double_outlier_local = set(range(len(main_outlier_indices))) - outlier_connected

        # Map outlier indices to global indices
        outlier_to_global = {i: main_outlier_indices[i] for i in range(len(main_outlier_indices))}

        if verbose:
            print(f"   Outlier ontology: {len(outlier_connected)} nodes, {sum(len(v) for v in outlier_ont.children.values())} edges")
            print(f"   Double outliers: {len(double_outlier_local)}")
    else:
        outlier_ont = None
        outlier_connected = set()
        double_outlier_local = set()
        outlier_to_global = {}

    # Step 3: Build unified structure
    if verbose:
        print("\n3. Merging into unified ontology...")

    children: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    parents: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    # Add main ontology edges
    for parent, child_list in main_ont.children.items():
        for child, sim, gap in child_list:
            children[parent].append((child, sim, gap))
            parents[child].append((parent, sim, gap))

    # Add outlier ontology edges (remapped to global indices)
    if outlier_ont is not None:
        for parent, child_list in outlier_ont.children.items():
            global_parent = outlier_to_global[parent]
            for child, sim, gap in child_list:
                global_child = outlier_to_global[child]
                children[global_parent].append((global_child, sim, gap))
                parents[global_child].append((global_parent, sim, gap))

    # Step 4: Add bridge edges
    if verbose:
        print("\n4. Adding bridge edges...")

    bridge_edges = 0
    if len(outlier_connected) > 0:
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
        nn.fit(embeddings)
        distances, neighbors = nn.kneighbors(embeddings)
        similarities = 1 - distances

        for outlier_local_idx in outlier_connected:
            global_idx = outlier_to_global[outlier_local_idx]
            for j in range(1, min(16, neighbors.shape[1])):
                neighbor = neighbors[global_idx, j]
                sim = similarities[global_idx, j]
                if neighbor in main_connected and sim >= outlier_similarity_threshold:
                    div_diff = main_ont.diversity[global_idx] - main_ont.diversity[neighbor]
                    if abs(div_diff) >= outlier_diversity_gap:
                        if div_diff > 0:  # outlier is parent
                            children[global_idx].append((neighbor, float(sim), float(div_diff)))
                            parents[neighbor].append((global_idx, float(sim), float(div_diff)))
                        else:  # main is parent
                            children[neighbor].append((global_idx, float(sim), float(-div_diff)))
                            parents[global_idx].append((neighbor, float(sim), float(-div_diff)))
                        bridge_edges += 1

    if verbose:
        print(f"   Bridge edges added: {bridge_edges}")

    # Convert to regular dicts
    children = dict(children)
    parents = dict(parents)

    # Find roots and leaves
    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    # Create unified ontology
    unified = DAGTaxonomy(
        n_nodes=n_points,
        diversity=main_ont.diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )

    # Compute result arrays
    main_nodes = np.array(sorted(main_connected), dtype=np.int64)
    outlier_nodes = np.array([outlier_to_global[i] for i in outlier_connected], dtype=np.int64)
    excluded_nodes = np.array([main_outlier_indices[i] for i in double_outlier_local], dtype=np.int64)

    if verbose:
        n_edges = sum(len(v) for v in children.values())
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        print(f"\n" + "="*60)
        print(f"Unified ontology complete:")
        print(f"  Coverage: {len(all_nodes)}/{n_points} ({100*len(all_nodes)/n_points:.1f}%)")
        print(f"  Edges: {n_edges} (main={sum(len(v) for v in main_ont.children.values())}, outlier={sum(len(v) for v in outlier_ont.children.values()) if outlier_ont else 0}, bridge={bridge_edges})")
        print(f"  Multi-parent: {n_multi} ({100*n_multi/max(len(all_nodes),1):.1f}%)")
        print(f"  Excluded: {len(excluded_nodes)} double outliers")

    return UnifiedOntologyResult(
        ontology=unified,
        main_nodes=main_nodes,
        outlier_nodes=outlier_nodes,
        excluded_nodes=excluded_nodes,
        bridge_edges=bridge_edges,
        main_threshold=main_similarity_threshold,
        outlier_threshold=outlier_similarity_threshold,
    )


# ============================================================
# ROG - Recursive Ontological Generation
# ============================================================


@dataclass
class ROGLayer:
    """
    A single layer in the ROG hierarchy.

    Attributes:
        depth: Recursion depth (0 = root layer)
        similarity_threshold: Threshold used for this layer
        node_indices: Global indices of nodes in this layer
        n_nodes: Number of nodes
        n_edges: Number of edges
        coverage: Fraction of input covered by this layer
    """
    depth: int
    similarity_threshold: float
    node_indices: np.ndarray
    n_nodes: int
    n_edges: int
    coverage: float


@dataclass
class ROGResult:
    """
    Result of Recursive Ontological Generation.

    ROG applies density-adaptive thresholding recursively:
    1. Build ontology at high threshold (dense regions)
    2. Recurse into outliers with lower threshold
    3. Repeat until coverage target or min threshold reached
    4. Knit layers together with bridge edges

    Attributes:
        ontology: The unified DAGTaxonomy
        layers: List of ROGLayer describing each recursion level
        excluded_nodes: Indices that couldn't be connected at any threshold
        total_coverage: Fraction of input nodes in ontology
        bridge_edges: Edges connecting different layers
    """
    ontology: DAGTaxonomy
    layers: List[ROGLayer]
    excluded_nodes: np.ndarray
    total_coverage: float
    bridge_edges: int

    def __len__(self) -> int:
        return len(self.ontology)

    def summary(self) -> str:
        """Return summary statistics."""
        lines = [
            f"ROGResult (Recursive Ontological Generation):",
            f"  Total coverage: {self.total_coverage*100:.1f}%",
            f"  Layers: {len(self.layers)}",
            f"  Bridge edges: {self.bridge_edges}",
            f"  Excluded: {len(self.excluded_nodes)}",
            f"",
            f"  Layer breakdown:",
        ]
        for layer in self.layers:
            lines.append(
                f"    Depth {layer.depth}: {layer.n_nodes} nodes, "
                f"{layer.n_edges} edges (sim≥{layer.similarity_threshold:.2f})"
            )
        return "\n".join(lines)

    def get_layer_for_node(self, node_idx: int) -> Optional[int]:
        """Return which layer a node belongs to, or None if excluded."""
        for layer in self.layers:
            if node_idx in set(layer.node_indices):
                return layer.depth
        return None


def build_rog_ontology(
    embeddings: np.ndarray,
    initial_threshold: float = 0.55,
    min_threshold: float = 0.35,
    threshold_decay: float = 0.9,
    target_coverage: float = 0.95,
    diversity_gap_threshold: float = 0.02,
    k_neighbors: int = 30,
    max_depth: int = 5,
    verbose: bool = False,
) -> ROGResult:
    """
    Recursive Ontological Generation (ROG).

    Recursively builds ontology layers at decreasing similarity thresholds,
    then knits them together. Achieves higher coverage than single-threshold
    approaches by adapting to local density.

    Algorithm:
        1. Build ontology at initial_threshold
        2. Identify outliers (unconnected nodes)
        3. If coverage < target and threshold > min:
           - Recurse on outliers with threshold * decay
        4. Knit all layers with bridge edges
        5. Return unified ontology with layer metadata

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        initial_threshold: Starting similarity threshold (default 0.55).
        min_threshold: Minimum threshold to try (default 0.35).
        threshold_decay: Multiply threshold by this each recursion (default 0.9).
        target_coverage: Stop recursing when this coverage reached (default 0.95).
        diversity_gap_threshold: Minimum diversity gap for edges (default 0.02).
        k_neighbors: Number of neighbors for k-NN graph.
        max_depth: Maximum recursion depth (default 5).
        verbose: Print progress information.

    Returns:
        ROGResult with unified ontology and layer metadata.

    Example:
        >>> from dyf import build_rog_ontology
        >>>
        >>> result = build_rog_ontology(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Access unified ontology
        >>> ontology = result.ontology
        >>> ancestors = ontology.get_ancestors(node_idx)
        >>>
        >>> # Check which layer a node came from
        >>> layer = result.get_layer_for_node(node_idx)
        >>> if layer is not None:
        ...     print(f"Node from layer {layer}")
    """
    from sklearn.neighbors import NearestNeighbors

    # Normalize embeddings
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    n_points, dim = embeddings.shape

    if verbose:
        print(f"ROG: Recursive Ontological Generation")
        print(f"  Points: {n_points:,}, dim={dim}")
        print(f"  Target coverage: {target_coverage*100:.0f}%")
        print(f"  Threshold range: {initial_threshold:.2f} → {min_threshold:.2f}")

    # Build k-NN once for the full dataset
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    # Compute diversity once for full dataset
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    # Recursive layer building
    layers: List[ROGLayer] = []
    all_children: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    all_parents: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    remaining_indices = set(range(n_points))
    threshold = initial_threshold
    depth = 0

    while (
        len(remaining_indices) / n_points > (1 - target_coverage)
        and threshold >= min_threshold
        and depth < max_depth
        and len(remaining_indices) > 10
    ):
        if verbose:
            print(f"\n  Layer {depth}: threshold={threshold:.3f}, candidates={len(remaining_indices)}")

        # Build ontology for remaining nodes at current threshold
        remaining_list = list(remaining_indices)
        layer_children: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        layer_parents: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        # Find edges among remaining nodes
        remaining_set = set(remaining_list)
        for i in remaining_list:
            for j_idx in range(1, min(16, neighbors.shape[1])):
                j = neighbors[i, j_idx]
                if j not in remaining_set:
                    continue

                sim = similarities[i, j_idx]
                if sim < threshold:
                    continue

                div_i = diversity[i]
                div_j = diversity[j]
                gap = div_i - div_j

                # Adjust gap threshold for deeper layers
                gap_thresh = diversity_gap_threshold * (0.9 ** depth)

                if gap > gap_thresh:
                    layer_children[i].append((j, float(sim), float(gap)))
                    layer_parents[j].append((i, float(sim), float(gap)))
                elif gap < -gap_thresh:
                    layer_children[j].append((i, float(sim), float(-gap)))
                    layer_parents[i].append((j, float(sim), float(-gap)))

        # Identify connected nodes in this layer
        connected = set(layer_children.keys()) | set(layer_parents.keys())
        n_edges = sum(len(v) for v in layer_children.values())

        if len(connected) > 0:
            # Add to global structure
            for parent, children in layer_children.items():
                all_children[parent].extend(children)
            for child, parents in layer_parents.items():
                all_parents[child].extend(parents)

            # Record layer
            layers.append(ROGLayer(
                depth=depth,
                similarity_threshold=threshold,
                node_indices=np.array(sorted(connected), dtype=np.int64),
                n_nodes=len(connected),
                n_edges=n_edges,
                coverage=len(connected) / n_points,
            ))

            if verbose:
                total_connected = len(set(all_children.keys()) | set(all_parents.keys()))
                print(f"    Connected: {len(connected)}, edges: {n_edges}")
                print(f"    Total coverage: {total_connected}/{n_points} ({100*total_connected/n_points:.1f}%)")

            # Update remaining
            remaining_indices -= connected

        # Decay threshold for next iteration
        threshold *= threshold_decay
        depth += 1

    # Add bridge edges between layers
    if verbose:
        print(f"\n  Adding bridge edges...")

    bridge_edges = 0
    all_connected = set(all_children.keys()) | set(all_parents.keys())

    for i in all_connected:
        for j_idx in range(1, min(16, neighbors.shape[1])):
            j = neighbors[i, j_idx]
            if j not in all_connected or j == i:
                continue

            # Check if edge already exists
            existing_children = {c for c, _, _ in all_children.get(i, [])}
            existing_parents = {p for p, _, _ in all_parents.get(i, [])}
            if j in existing_children or j in existing_parents:
                continue

            sim = similarities[i, j_idx]
            if sim < min_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            if abs(gap) > diversity_gap_threshold * 0.5:  # Looser for bridges
                if gap > 0:
                    all_children[i].append((j, float(sim), float(gap)))
                    all_parents[j].append((i, float(sim), float(gap)))
                else:
                    all_children[j].append((i, float(sim), float(-gap)))
                    all_parents[i].append((j, float(sim), float(-gap)))
                bridge_edges += 1

    if verbose:
        print(f"    Bridge edges: {bridge_edges}")

    # Build final ontology
    children = dict(all_children)
    parents = dict(all_parents)

    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    ontology = DAGTaxonomy(
        n_nodes=n_points,
        diversity=diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )

    excluded = np.array(sorted(remaining_indices), dtype=np.int64)
    total_coverage = len(all_nodes) / n_points

    if verbose:
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        print(f"\n  ROG complete:")
        print(f"    Layers: {len(layers)}")
        print(f"    Coverage: {len(all_nodes)}/{n_points} ({total_coverage*100:.1f}%)")
        print(f"    Excluded: {len(excluded)}")
        print(f"    Multi-parent: {n_multi} ({100*n_multi/max(len(all_nodes),1):.1f}%)")

    return ROGResult(
        ontology=ontology,
        layers=layers,
        excluded_nodes=excluded,
        total_coverage=total_coverage,
        bridge_edges=bridge_edges,
    )
