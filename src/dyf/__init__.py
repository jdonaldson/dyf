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
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    DensityClassifier = None
    DensityReport = None
    BridgeAnalysis = None

# Embedding and labeling configs
from .configs import (
    EmbedderConfig,
    LabelerConfig,
    list_configs,
)

# Python wrapper with full features
from .classifier import (
    DensityClassifier as DensityClassifierFull,
)

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

# DYF tree (recursive k-ary LSH splits)
from .dyf_tree import (
    build_dyf_tree,
    refine_dyf_tree,
    cut_dyf_tree_to_labels,
    refine_clusters,
)
# Also available: dyf_tree.extract_boundary_persistence,
#                 dyf_tree.boundary_persistence_scores

# Lazy index (FlatBuffers + Arrow IPC)
try:
    from .lazy_index import (
        LazyIndex, write_lazy_index, rewrite_lazy_index,
        from_faiss, SearchResult,
    )
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False
    LazyIndex = None
    write_lazy_index = None
    rewrite_lazy_index = None
    from_faiss = None
    SearchResult = None

# Fisher dimension weighting
from .fisher import (
    compute_fisher_weights,
    apply_fisher_weights,
    extract_fisher_labels,
)

# Categorical DAG
from .categorical import (
    AxisDiagnostic,
    CategoryGraph,
    coarsen,
    diagnose_axes,
    diagnostics_to_metadata,
    discover_categorical_columns,
    embed_with_diagnostics,
    multi_level_fisher_weights,
    store_category_graph,
    load_category_graphs,
)

# CatalogSpace (multi-catalog matching)
from .catalog import (
    CatalogConfig,
    CatalogMatch,
    CatalogSpace,
    CrossMapping,
    JointMatchResult,
)

# Split-based tree keywords
from .splits import (
    TextDiversityReport,
    assess_text_diversity,
    build_tree_maps,
    collect_descendant_indices,
    compute_domain_stopwords,
    compute_embedding_keywords,
    compute_split_keywords,
    format_split_path,
    label_clusters_frequency,
    tokenize,
)

# Cluster-tree DAG
from .cluster_tree import (
    build_cluster_tree_dag,
    compute_sibling_keywords,
    derive_path_labels,
    format_cluster_context,
)

# Tree-leaf agglomeration
from .agglomerate import agglomerate_tree_leaves, louvain_cluster_leaves, merge_to_max_k

# Spatial cluster coloring
from .colors import spatial_rgb_map, spatial_color_map, tree_rgb_map

# Pipeline DAG runner
from .pipeline import Pipeline, Stage

# Provenance tracking
from .provenance import (
    Provenance,
    file_hash,
    params_hash,
    create_provenance,
    check_compatible,
    provenance_to_dict,
    provenance_from_dict,
)

# Concept graph
from .concept_graph import (
    ConceptGraphConfig,
    ConceptNode,
    MarkdownChunk,
    build_concept_graph,
    chunk_markdown,
    fuzzy_match,
    load_graph,
    save_graph,
    semantic_search,
    check_staleness,
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
    find_super_connectors,
    select_orthogonal_anchors,
    diversify_by_facet,
    get_kmeans_init,
)

# Ontology (DAG taxonomy extraction)
from .ontology import (
    DAGChain,
    DAGMiningResult,
    DAGTaxonomy,
    UnifiedOntologyResult,
    ROGLayer,
    ROGResult,
    HubScoreResult,
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
    # Full Python wrapper
    "DensityClassifierFull",
    "EmbedderConfig",
    "LabelerConfig",
    "list_configs",
    # RAG index
    "BridgeIndex",
    "SuperConnectorResult",
    "OrthogonalAnchorResult",
    "FacetDiverseResult",
    "find_super_connectors",
    "select_orthogonal_anchors",
    "diversify_by_facet",
    "get_kmeans_init",
    # Ontology
    "DAGChain",
    "DAGMiningResult",
    "DAGTaxonomy",
    "UnifiedOntologyResult",
    "ROGLayer",
    "ROGResult",
    "HubScoreResult",
    "compute_neighbor_diversity",
    "compute_hub_score",
    "mine_dag_chains",
    "build_dag_taxonomy",
    "build_unified_ontology",
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
    # DYF tree
    "build_dyf_tree",
    "refine_dyf_tree",
    "cut_dyf_tree_to_labels",
    "refine_clusters",
    # Lazy index
    "LazyIndex",
    "write_lazy_index",
    "rewrite_lazy_index",
    "from_faiss",
    "SearchResult",
    # Fisher dimension weighting
    "compute_fisher_weights",
    "apply_fisher_weights",
    "extract_fisher_labels",
    # Categorical DAG
    "AxisDiagnostic",
    "CategoryGraph",
    "coarsen",
    "diagnose_axes",
    "diagnostics_to_metadata",
    "discover_categorical_columns",
    "embed_with_diagnostics",
    "multi_level_fisher_weights",
    "store_category_graph",
    "load_category_graphs",
    # CatalogSpace
    "CatalogConfig",
    "CatalogMatch",
    "CatalogSpace",
    "CrossMapping",
    "JointMatchResult",
    # Split-based tree keywords
    "TextDiversityReport",
    "assess_text_diversity",
    "build_tree_maps",
    "collect_descendant_indices",
    "compute_domain_stopwords",
    "compute_embedding_keywords",
    "compute_split_keywords",
    "format_split_path",
    "label_clusters_frequency",
    "tokenize",
    # Cluster-tree DAG
    "build_cluster_tree_dag",
    "compute_sibling_keywords",
    "derive_path_labels",
    "format_cluster_context",
    # Tree-leaf agglomeration
    "agglomerate_tree_leaves",
    "louvain_cluster_leaves",
    "merge_to_max_k",
    # Spatial cluster coloring
    "spatial_rgb_map",
    "spatial_color_map",
    # Pipeline DAG runner
    "Pipeline",
    "Stage",
    # Provenance tracking
    "Provenance",
    "file_hash",
    "params_hash",
    "create_provenance",
    "check_compatible",
    "provenance_to_dict",
    "provenance_from_dict",
    # Re-ranking
    "rerank_standard",
    "rerank_mmr",
    "rerank_bridge_boost",
    "rerank_bridge_mmr",
    # Concept graph
    "ConceptGraphConfig",
    "ConceptNode",
    "MarkdownChunk",
    "build_concept_graph",
    "chunk_markdown",
    "fuzzy_match",
    "load_graph",
    "save_graph",
    "semantic_search",
    "check_staleness",
]

def check_rust_available():
    """Check if Rust acceleration is available."""
    return _HAS_RUST
