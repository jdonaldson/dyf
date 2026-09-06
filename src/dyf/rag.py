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

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ._arrays import ensure_f32
from .lazy_index import SearchResult

logger = logging.getLogger(__name__)

try:
    from dyf_rs import BridgeAnalysis, DensityClassifier  # noqa: F401

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
        n_cross = (self.quadrant == "Cross-Domain").sum()
        n_specialist = (self.quadrant == "Domain Specialist").sum()
        n_minor = (self.quadrant == "Minor Bridge").sum()
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


#: Default percentile of a corpus's own centroid-similarity distribution used as the bridge
#: cut. See :func:`_relative_bridge_threshold` for why this is not an absolute constant.
DEFAULT_BRIDGE_PERCENTILE = 10.0


def _derive_num_bits(n_points: int, min_bucket_size: int, cap: int = 12) -> int:
    """Bucket resolution scaled to corpus size, so dense buckets can actually exist.

    Faceting only runs inside buckets that clear ``min_bucket_size``. A *fixed* bit count
    fixes the bucket count at ``2**bits`` regardless of ``n``, so below some corpus size no
    bucket ever qualifies and the dense-bucket gate closes completely. Measured with the old
    ``global_num_bits=12`` default (4096 buckets, ``min_bucket_size=20``):

    =========  =======  ==============  =============  ================
    data       n        largest bucket  dense buckets  super connectors
    =========  =======  ==============  =============  ================
    isotropic      500               2              0                 0
    isotropic   30,000              20              0                 0
    clustered    2,000              18              0                 0
    clustered    8,000             113            112                50
    =========  =======  ==============  =============  ================

    So the feature was silently inert below ~8k points, and on isotropic data at every size
    tested. Targeting a mean occupancy of ``2 * min_bucket_size`` puts the upper quartile of
    buckets clear of the gate at any ``n``; the cap preserves the previous behaviour on large
    corpora, where 12 bits was already in the right regime.

    This is the same failure as an absolute cosine threshold (see
    :func:`_relative_bridge_threshold`) with *corpus size* as the axis that does not
    transfer rather than anisotropy.
    """
    if n_points <= 0:
        return 2
    target_buckets = max(n_points / max(2 * min_bucket_size, 1), 4.0)
    return int(np.clip(int(np.log2(target_buckets)), 2, cap))


def _relative_bridge_threshold(clf, percentile: float = DEFAULT_BRIDGE_PERCENTILE) -> float:
    """Bridge cut as a percentile of *this* corpus's centroid-similarity distribution.

    `DensityClassifier.analyze_bridges` treats a point as a bridge when its centroid
    similarity falls below `bridge_threshold`, whose 0.5 default is an **absolute cosine**.
    Embedding anisotropy varies enormously, so no constant transfers. Measured over
    4,000-point samples:

    ==========================  ==================  ================
    corpus                      % below cosine 0.5  bridges flagged
    ==========================  ==================  ================
    SEC 768d unit-norm text                   0.0%                0
    CMU MoCap 62d                             0.4%               14
    isotropic gaussian 64d                   92.3%    3,693 of 4,000
    ==========================  ==================  ================

    So the default finds nothing or nearly everything. A percentile lands in a usable regime
    on every corpus. Every `analyze_bridges` call in this module routes through here; passing
    ``percentile=0`` reproduces the old absolute-floor behaviour for comparison.
    """
    cs = np.asarray(clf.get_centroid_similarities(), dtype=np.float64)
    if len(cs) == 0:
        return 0.5
    return float(np.percentile(cs, percentile))


def _precompute_neighborhoods(
    anchor_indices: np.ndarray,
    embeddings: np.ndarray,
    expansion_k: int,
) -> dict[int, np.ndarray]:
    """
    Precompute k-nearest-neighbor neighborhoods for each anchor point.

    For each anchor, computes cosine similarities against all embeddings
    and stores the top-k nearest neighbor indices.

    Args:
        anchor_indices: Array of anchor point indices
        embeddings: Normalized embeddings (n_points, dim)
        expansion_k: Number of nearest neighbors per anchor

    Returns:
        Dict mapping each anchor index to its top-k neighbor indices
    """
    neighborhoods: dict[int, np.ndarray] = {}
    for anchor_idx in anchor_indices:
        sims = embeddings @ embeddings[anchor_idx]
        top_k = np.argsort(sims)[-expansion_k:][::-1]
        neighborhoods[anchor_idx] = top_k
    return neighborhoods


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
        global_num_bits: LSH bits for global bucketing (default None = derive from
            corpus size; a fixed value closes the dense-bucket gate on small corpora)
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
    global_num_bits: int | None = None
    facet_num_bits: int | None = None
    dense_percentile: float = 75
    min_bucket_size: int = 20
    include_sparse_points: int = 0  # Number of sparse region points to add
    seed: int = 42

    # Fitted state (populated by fit())
    _embeddings: np.ndarray | None = field(default=None, repr=False)
    _anchor_indices: np.ndarray | None = field(default=None, repr=False)
    _anchor_embeddings: np.ndarray | None = field(default=None, repr=False)
    _neighborhoods: dict[int, np.ndarray] | None = field(default=None, repr=False)
    _super_connectors: SuperConnectorResult | None = field(default=None, repr=False)
    _bridge_indices: np.ndarray | None = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(self, embeddings: np.ndarray, verbose: bool = True) -> "BridgeIndex":
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

        # Resolve bucket resolution against the corpus we were actually handed, and record
        # it, so `global_num_bits` reads back as the value that was used rather than None.
        if self.global_num_bits is None:
            self.global_num_bits = _derive_num_bits(n_points, self.min_bucket_size)
        if self.facet_num_bits is None:
            self.facet_num_bits = max(2, self.global_num_bits - 2)

        if verbose:
            logger.info(
                f"Building BridgeIndex: {n_points:,} points, dim={dim}, "
                f"num_bits={self.global_num_bits}/{self.facet_num_bits}"
            )

        # Step 1: Find super connectors
        if verbose:
            logger.info("  Finding super connectors...")
        self._super_connectors = find_super_connectors(
            embeddings,
            global_num_bits=self.global_num_bits,
            facet_num_bits=self.facet_num_bits,
            dense_percentile=self.dense_percentile,
            min_bucket_size=self.min_bucket_size,
            seed=self.seed,
        )

        if verbose:
            logger.info(f"    {self._super_connectors.summary()}")

        # Step 2: Get all bridge indices
        clf = DensityClassifier(embedding_dim=dim, num_bits=self.global_num_bits, seed=self.seed)
        clf.fit(embeddings)
        bridge_analysis = clf.analyze_bridges(embeddings, bridge_threshold=_relative_bridge_threshold(clf))
        self._bridge_indices = np.array(bridge_analysis.bridge_indices)

        if verbose:
            logger.info(f"    Total bridges: {len(self._bridge_indices):,}")

        # Step 3: Select orthogonal anchors
        if verbose:
            logger.info(f"  Selecting {self.n_anchors} orthogonal anchors...")

        # If we have bridges, use them as candidates; otherwise fall back to all points
        if len(self._bridge_indices) > 0:
            candidate_indices = self._bridge_indices
        else:
            if verbose:
                logger.warning("    No bridges found, using all points as candidates")
            candidate_indices = None  # Will default to all points

        # If we have super connectors, use them as seeds; otherwise use None
        seed_indices = self._super_connectors.indices if len(self._super_connectors.indices) > 0 else None

        anchor_result = select_orthogonal_anchors(
            embeddings,
            k=self.n_anchors - self.include_sparse_points,
            seed_indices=seed_indices,
            candidate_indices=candidate_indices,
            use_bridges=False,  # We already handled bridge detection above
            seed=self.seed,
        )

        anchor_indices = list(anchor_result.indices)

        # Step 4: Add sparse region points if requested
        if self.include_sparse_points > 0:
            if verbose:
                logger.info(f"  Adding {self.include_sparse_points} sparse region points...")

            bucket_sizes = clf.get_bucket_sizes()
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
            logger.info(f"    Final anchors: {len(self._anchor_indices):,}")

        # Step 5: Precompute neighborhoods for each anchor
        if verbose:
            logger.info(f"  Precomputing {self.expansion_k}-NN neighborhoods...")

        self._neighborhoods = _precompute_neighborhoods(self._anchor_indices, embeddings, self.expansion_k)

        self._fitted = True

        if verbose:
            storage_mb = self._estimate_storage_mb()
            logger.info(f"  Index built: {storage_mb:.1f} MB")

        return self

    def query(
        self, query: np.ndarray, k: int = 10, n_query_anchors: int | None = None, return_scores: bool = True
    ) -> SearchResult:
        """
        Retrieve top-k candidates for a query embedding.

        Returns the same :class:`SearchResult` as `LazyIndex.search` and
        `DenseSearchIndex.search`, so every retriever in the package answers with one
        shape. It unpacks as a 2-tuple, so `indices, scores = index.query(...)` keeps
        working.

        Args:
            query: (d,) query embedding vector
            k: Number of results to return
            n_query_anchors: Override default number of anchors to probe
            return_scores: Whether to compute similarity scores (`scores` is None if not)

        Returns:
            SearchResult whose `indices` are the top-k candidate indices
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

        top_k_scores = candidate_sims[top_k_local] if return_scores else None
        return SearchResult(indices=top_k_indices, scores=top_k_scores)

    def query_batch(self, queries: np.ndarray, k: int = 10, n_query_anchors: int | None = None) -> list[SearchResult]:
        """
        Batch query for multiple embeddings.

        Args:
            queries: (n_queries, d) array of query embeddings
            k: Number of results per query
            n_query_anchors: Override default number of anchors to probe

        Returns:
            List of SearchResult, one per query. Each unpacks as (indices, scores).
        """
        return [self.query(query, k=k, n_query_anchors=n_query_anchors) for query in queries]

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
        neighborhood_bytes = sum(arr.nbytes for arr in self._neighborhoods.values())

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
            f"  Anchors: {n_anchors:,} ({n_anchors / n_points * 100:.1f}%)\n"
            f"  Super connectors: {n_super:,}\n"
            f"  Total bridges: {n_bridges:,}\n"
            f"  Avg neighborhood: {avg_neighborhood:.0f}\n"
            f"  Storage: {storage_mb:.1f} MB"
        )

    def evaluate_recall(self, n_queries: int = 100, k: int = 10, seed: int = 42) -> dict[str, float]:
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
            top_anchors = self._anchor_indices[np.argsort(anchor_sims)[-self.n_query_anchors :]]
            candidates = set()
            for anchor in top_anchors:
                candidates.update(self._neighborhoods[anchor])
            total_candidates += len(candidates)

        avg_recall = total_recall / n_queries
        avg_candidates = total_candidates / n_queries
        speedup = len(self._embeddings) / avg_candidates if avg_candidates > 0 else 0

        return {"recall": avg_recall, "avg_candidates": avg_candidates, "speedup": speedup}


def _compute_local_centrality(
    embeddings: np.ndarray,
    bucket_to_indices: dict[int, list[int]],
    dense_bucket_ids: np.ndarray,
    n_points: int,
    dim: int,
    facet_num_bits: int,
    min_bucket_size: int,
    seed: int,
    bridge_percentile: float = 10,
) -> np.ndarray:
    """
    Compute local (facet) bridge centrality within dense buckets.

    For each dense bucket, runs a sub-LSH bridge analysis and counts
    how many facets each point connects. Points that bridge multiple
    facets within a dense region get higher local centrality scores.

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        bucket_to_indices: Mapping from bucket ID to list of point indices
        dense_bucket_ids: Array of bucket IDs considered dense
        n_points: Total number of points
        dim: Embedding dimensionality
        facet_num_bits: LSH bits for facet bucketing
        min_bucket_size: Minimum bucket size for faceting
        seed: Random seed

    Returns:
        Array of local centrality scores (n_points,)
    """
    local_centrality = np.zeros(n_points, dtype=np.int32)

    for bid in dense_bucket_ids:
        indices = np.array(bucket_to_indices[bid])
        if len(indices) < min_bucket_size:
            continue

        bucket_emb = embeddings[indices]
        bits = 6 if len(indices) < 100 else (8 if len(indices) < 500 else facet_num_bits)

        try:
            facet_clf = DensityClassifier(embedding_dim=dim, num_bits=bits, seed=seed)
            facet_clf.fit(ensure_f32(bucket_emb, "embeddings"))
            facet_bridge = facet_clf.analyze_bridges(
                bucket_emb, bridge_threshold=_relative_bridge_threshold(facet_clf, bridge_percentile)
            )

            for i in range(len(facet_bridge.bridge_indices)):
                local_idx, _, neighbors = facet_bridge.get_bridge_connections(i)
                local_centrality[indices[local_idx]] = len(neighbors) + 1
        except TypeError:
            raise  # dtype/signature errors are bugs, never graceful-degradation
        except Exception as e:
            logger.debug("Facet bridge analysis failed for bucket: %s", e)

    return local_centrality


def find_super_connectors(
    embeddings: np.ndarray,
    global_num_bits: int | None = None,
    facet_num_bits: int | None = None,
    dense_percentile: float = 75,
    global_threshold_percentile: float = 50,
    local_threshold_percentile: float = 50,
    min_bucket_size: int = 20,
    seed: int = 42,
    bridge_percentile: float = 10,
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
        global_num_bits: LSH bits for global bucketing. ``None`` (default) derives it from
            the corpus size via :func:`_derive_num_bits`, which is required for the
            dense-bucket gate to open at all below ~8k points — the previous fixed default
            of 12 returned zero super connectors on every smaller corpus.
        facet_num_bits: LSH bits for facet bucketing within a dense bucket. ``None``
            (default) uses two bits below the global resolution.
        dense_percentile: Percentile threshold for dense buckets
        global_threshold_percentile: Percentile for "high" global centrality
        local_threshold_percentile: Percentile for "high" local centrality
        min_bucket_size: Minimum bucket size for faceting
        seed: Random seed
        bridge_percentile: Bridges are the lowest-centroid-similarity points; this is the
            percentile of *this corpus's* similarity distribution used as the cut. Relative
            by design — `analyze_bridges`'s absolute 0.5 default returns zero bridges on
            unit-norm text embeddings (0.0% fall below it) and floods on isotropic data
            (92.3% below), so an absolute constant cannot serve both.

    Returns:
        SuperConnectorResult with indices and centrality data
    """
    n_points, dim = embeddings.shape

    if global_num_bits is None:
        global_num_bits = _derive_num_bits(n_points, min_bucket_size)
    if facet_num_bits is None:
        facet_num_bits = max(2, global_num_bits - 2)

    # Global DYF and bridge analysis
    global_clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
    global_clf.fit(embeddings)
    global_buckets = global_clf.get_bucket_ids()

    global_bridge = global_clf.analyze_bridges(
        embeddings, bridge_threshold=_relative_bridge_threshold(global_clf, bridge_percentile)
    )

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
    local_centrality = _compute_local_centrality(
        embeddings,
        bucket_to_indices,
        dense_bucket_ids,
        n_points,
        dim,
        facet_num_bits,
        min_bucket_size,
        seed,
        bridge_percentile,
    )

    # Compute thresholds from non-zero values
    global_nonzero = global_centrality[global_centrality > 0]
    local_nonzero = local_centrality[local_centrality > 0]

    global_thresh = np.percentile(global_nonzero, global_threshold_percentile) if len(global_nonzero) > 0 else 1
    local_thresh = np.percentile(local_nonzero, local_threshold_percentile) if len(local_nonzero) > 0 else 1

    # Classify into quadrants.
    # `>=`, not `>`: centrality is a small integer count, so its percentile frequently lands
    # ON the modal value. Measured on 8k SEC points the 50th percentile of nonzero global
    # centrality equalled the maximum (195), so a strict `>` selected nothing and the
    # function returned an empty `indices` even once bridges were being found.
    high_global = global_centrality >= global_thresh
    high_local = local_centrality >= local_thresh
    quadrant = np.full(n_points, "Regular", dtype=object)
    is_bridge = (global_centrality > 0) | (local_centrality > 0)

    quadrant[is_bridge & ~high_global & ~high_local] = "Minor Bridge"
    quadrant[high_global & ~high_local] = "Cross-Domain"
    quadrant[~high_global & high_local] = "Domain Specialist"
    quadrant[high_global & high_local] = "Super Connector"

    super_indices = np.where(quadrant == "Super Connector")[0]

    return SuperConnectorResult(
        indices=super_indices,
        global_centrality=global_centrality,
        local_centrality=local_centrality,
        quadrant=quadrant,
        global_threshold=float(global_thresh),
        local_threshold=float(local_thresh),
    )


def select_orthogonal_anchors(
    embeddings: np.ndarray,
    k: int,
    seed_indices: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
    use_bridges: bool = True,
    global_num_bits: int = 12,
    seed: int = 42,
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
            bridge_analysis = clf.analyze_bridges(embeddings, bridge_threshold=_relative_bridge_threshold(clf))
            candidate_indices = np.array(bridge_analysis.bridge_indices)
            candidate_source = "bridges"
            # Fall back to all points if no bridges found. Before the threshold was made
            # relative this fired on EVERY unit-norm text corpus, so `use_bridges=True`
            # silently produced output identical to `use_bridges=False` — measured on 3k SEC
            # sections, 60 anchors either way, with no warning.
            if len(candidate_indices) == 0:
                logger.warning(
                    "use_bridges=True found no bridges; falling back to all %d points. "
                    "If this repeats, the corpus may need a larger bridge percentile.",
                    n_points,
                )
                candidate_indices = np.arange(n_points)
                candidate_source = "all"
        else:
            candidate_indices = np.arange(n_points)
            candidate_source = "all"
    else:
        candidate_source = "custom"
        # Handle empty custom candidates
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(n_points)
            candidate_source = "all"

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
        indices=np.array(selected), seed_indices=seed_indices, candidate_source=candidate_source
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
            buckets_covered=0,
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
        buckets_covered=len(seen_buckets),
    )


def _find_candidate_bridges(
    embeddings: np.ndarray,
    n_points: int,
    dim: int,
    use_stable_bridges: bool,
    num_stability_seeds: int,
    stability_threshold: int,
    global_num_bits: int,
    seed: int,
    nlist: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Find candidate bridge points via stable-multi-seed or single-seed detection.

    For stable bridges, runs LSH bridge detection across multiple seeds and
    keeps points that appear in at least ``stability_threshold`` seeds. Falls
    back to single-seed detection or all points if not enough candidates.

    Args:
        embeddings: Normalized embeddings (n_points, dim)
        n_points: Number of points
        dim: Embedding dimensionality
        use_stable_bridges: If True, prefer bridges stable across multiple seeds
        num_stability_seeds: Number of seeds for stability testing
        stability_threshold: Minimum seeds a bridge must appear in
        global_num_bits: LSH bits for bridge detection
        seed: Random seed
        nlist: Minimum number of candidates needed
        verbose: Print progress

    Returns:
        Array of candidate bridge indices
    """
    if use_stable_bridges and num_stability_seeds > 1:
        if verbose:
            logger.info(f"  Finding stable bridges ({num_stability_seeds} seeds)...")

        # Count bridge appearances across seeds
        bridge_counts = np.zeros(n_points, dtype=np.int32)

        for seed_idx in range(num_stability_seeds):
            seed_offset = seed_idx * 1000
            clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed + seed_offset)
            clf.fit(embeddings)
            bridge_analysis = clf.analyze_bridges(embeddings, bridge_threshold=_relative_bridge_threshold(clf))
            for bridge_idx in bridge_analysis.bridge_indices:
                bridge_counts[bridge_idx] += 1

        # Select stable bridges
        stable_mask = bridge_counts >= stability_threshold
        stable_bridges = np.where(stable_mask)[0]

        if verbose:
            total_bridges = (bridge_counts > 0).sum()
            logger.info(f"    Total bridges: {total_bridges}, stable: {len(stable_bridges)}")

        if len(stable_bridges) >= nlist:
            candidate_indices = stable_bridges
        else:
            # Fall back to all bridges
            candidate_indices = np.where(bridge_counts > 0)[0]
    else:
        # Use single-seed bridges
        clf = DensityClassifier(embedding_dim=dim, num_bits=global_num_bits, seed=seed)
        clf.fit(embeddings)
        bridge_analysis = clf.analyze_bridges(embeddings, bridge_threshold=_relative_bridge_threshold(clf))
        candidate_indices = np.array(bridge_analysis.bridge_indices)

        if verbose:
            logger.info(f"  Found {len(candidate_indices)} bridges")

    # Fall back to all points if not enough candidates
    if len(candidate_indices) < nlist:
        if verbose:
            logger.warning(f"  Not enough candidates ({len(candidate_indices)}), using all points")
        candidate_indices = np.arange(n_points)

    return candidate_indices


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
        logger.info(f"Computing bridge-seeded k-means initialization (nlist={nlist})...")

    # Find candidate bridges
    candidate_indices = _find_candidate_bridges(
        embeddings,
        n_points,
        dim,
        use_stable_bridges=use_stable_bridges,
        num_stability_seeds=num_stability_seeds,
        stability_threshold=stability_threshold,
        global_num_bits=global_num_bits,
        seed=seed,
        nlist=nlist,
        verbose=verbose,
    )

    # Select orthogonal anchors
    if verbose:
        logger.info(f"  Selecting {nlist} orthogonal anchors...")

    anchor_result = select_orthogonal_anchors(
        embeddings,
        k=nlist,
        candidate_indices=candidate_indices,
        use_bridges=False,  # Already handled above
        seed=seed,
    )

    # Return anchor embeddings as initial centroids
    init_centroids = embeddings[anchor_result.indices].copy()

    # Pad if we didn't get enough anchors
    if len(init_centroids) < nlist:
        if verbose:
            logger.warning(f"  Padding from {len(init_centroids)} to {nlist} centroids")
        rng = np.random.default_rng(seed)
        remaining = nlist - len(init_centroids)
        existing_set = set(anchor_result.indices)
        available = [i for i in range(n_points) if i not in existing_set]
        extra = rng.choice(available, min(remaining, len(available)), replace=False)
        init_centroids = np.vstack([init_centroids, embeddings[extra]])

    if verbose:
        logger.info(f"  Done. Returned {len(init_centroids)} initial centroids")

    return init_centroids.astype(np.float32)
