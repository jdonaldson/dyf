"""
DYF - Density Yields Features

Discover structure in embedding spaces using PCA-based LSH.

The Rust core returns raw density metrics per item - classification is up to you:
- bucket_id: LSH bucket assignment
- bucket_size: Number of items in the bucket
- centroid_similarity: Cosine similarity to bucket centroid (0-1)
- isolation_score: How isolated the item is (top_k_sim - median_sim)

Quick Start:
    >>> from dyf import DensityClassifier
    >>> classifier = DensityClassifier(embedding_dim=384)
    >>> classifier.fit(embeddings)
    >>> print(classifier.report())
    >>> bucket_sizes = classifier.get_bucket_sizes()
    >>> isolation_scores = classifier.get_isolation_scores()

Full-Featured Usage:
    >>> from dyf import DensityClassifierFull, EmbedderConfig, LabelerConfig
    >>> classifier = DensityClassifierFull.from_texts(texts, categories=categories)
    >>> labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
"""

# Fast Rust implementation (core classifier)
try:
    from dyf_rs import (
        DensityClassifier,
        DensityReport,
        BridgeAnalysis,
        BridgePersistence,
        MultiResolutionAnalysis,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    DensityClassifier = None
    DensityReport = None
    BridgeAnalysis = None
    BridgePersistence = None
    MultiResolutionAnalysis = None

# Python wrapper with full features (embedder configs, labeling, etc.)
from .classifier import (
    DensityClassifier as DensityClassifierFull,
    DensityReport as DensityReportFull,
    EmbedderConfig,
    LabelerConfig,
    list_configs,
)

# Index serialization
from .io import save_index, load_index, PrecomputedIndex

# Chunk analysis
from .chunks import (
    DocSpread,
    chunk_redundancy,
    cluster_quality,
    deduplicate_chunks,
    doc_spread,
    neighbor_coherence,
)

# PCA tree
from .pca_tree import (
    build_pca_tree,
    extract_boundary_persistence,
    boundary_persistence_scores,
    cut_tree_to_labels,
)

# Re-ranking
from .rerank import (
    rerank_standard,
    rerank_mmr,
    rerank_bridge_boost,
    rerank_bridge_mmr,
)

# RAG index
from .rag import (
    BridgeIndex,
    SuperConnectorResult,
    OrthogonalAnchorResult,
    FacetDiverseResult,
    DAGChain,
    DAGMiningResult,
    DAGTaxonomy,
    UnifiedOntologyResult,
    ROGLayer,
    ROGResult,
    HubScoreResult,
    find_super_connectors,
    select_orthogonal_anchors,
    diversify_by_facet,
    get_kmeans_init,
    compute_neighbor_diversity,
    compute_hub_score,
    mine_dag_chains,
    build_dag_taxonomy,
    build_unified_ontology,
    build_rog_ontology,
)

from importlib.metadata import version as _get_version
__version__ = _get_version("dyf")
__all__ = [
    # Fast Rust core
    "DensityClassifier",
    "DensityReport",
    "BridgeAnalysis",
    "BridgePersistence",
    "MultiResolutionAnalysis",
    # Full Python wrapper
    "DensityClassifierFull",
    "DensityReportFull",
    "EmbedderConfig",
    "LabelerConfig",
    "list_configs",
    # Serialization
    "save_index",
    "load_index",
    "PrecomputedIndex",
    # RAG index
    "BridgeIndex",
    "SuperConnectorResult",
    "OrthogonalAnchorResult",
    "FacetDiverseResult",
    "DAGChain",
    "DAGMiningResult",
    "find_super_connectors",
    "select_orthogonal_anchors",
    "diversify_by_facet",
    "get_kmeans_init",
    "compute_neighbor_diversity",
    "compute_hub_score",
    "HubScoreResult",
    "mine_dag_chains",
    "DAGTaxonomy",
    "build_dag_taxonomy",
    "UnifiedOntologyResult",
    "build_unified_ontology",
    "ROGLayer",
    "ROGResult",
    "build_rog_ontology",
    # Chunk analysis
    "DocSpread",
    "chunk_redundancy",
    "cluster_quality",
    "deduplicate_chunks",
    "doc_spread",
    "neighbor_coherence",
    # PCA tree
    "build_pca_tree",
    "extract_boundary_persistence",
    "boundary_persistence_scores",
    "cut_tree_to_labels",
    # Re-ranking
    "rerank_standard",
    "rerank_mmr",
    "rerank_bridge_boost",
    "rerank_bridge_mmr",
]

def check_rust_available():
    """Check if Rust acceleration is available."""
    return _HAS_RUST
