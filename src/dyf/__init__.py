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
        BridgeAnalysis,
        DensityClassifier,
        DensityReport,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    DensityClassifier = None
    DensityReport = None
    BridgeAnalysis = None

# Warn if dyf-rs is installed but below the documented floor.
# The pyproject constraint (dyf-rs>=0.7.0) can be silently bypassed
# by stale editable-install metadata.
if _HAS_RUST:
    import warnings as _warnings
    _DYF_RS_FLOOR = (0, 7, 0)
    _rs_ver_str = getattr(__import__("dyf_rs"), "__version__", None)
    if _rs_ver_str:
        try:
            _rs_ver = tuple(int(x) for x in _rs_ver_str.split(".")[:3])
            if _rs_ver < _DYF_RS_FLOOR:
                _warnings.warn(
                    f"dyf-rs {_rs_ver_str} is below the documented floor "
                    f"{'.'.join(str(x) for x in _DYF_RS_FLOOR)}. "
                    f"Run `uv pip install -e .` to refresh editable-install metadata.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (ValueError, TypeError):
            pass
    del _warnings, _DYF_RS_FLOOR, _rs_ver_str

# Embedding and labeling configs
# Chunk analysis
from .chunks import (
    DocSpread,
    chunk_redundancy,
    cluster_quality,
    deduplicate_chunks,
    doc_spread,
    neighbor_coherence,
)

# Python wrapper with full features
from .classifier import (
    DensityClassifier as DensityClassifierFull,
)
from .configs import (
    EmbedderConfig,
    LabelerConfig,
    list_configs,
)

# DYF tree (recursive k-ary LSH splits)
from .dyf_tree import (
    build_dyf_tree,
    refine_clusters,
    refine_dyf_tree,
)

# PCA tree
from .pca_tree import (
    boundary_persistence_scores,
    build_pca_tree,
    extract_boundary_persistence,
)

# Unified tree-cutting dispatcher (routes to the right impl per tree shape)
from .cut import cut_tree_to_labels

# Also available: dyf_tree.extract_boundary_persistence,
#                 dyf_tree.boundary_persistence_scores

# Lazy index (FlatBuffers + Arrow IPC)
try:
    from .lazy_index import (
        AdaptiveProbeConfig,
        ExtractedData,
        LazyIndex,
        SearchResult,
        StoredFieldInput,
        StoredFieldValue,
        TreeNode,
        detect_dyf_version,
        from_faiss,
        rewrite_lazy_index,
        split_dyf3,
        write_lazy_index,
    )
    _HAS_LAZY = True
except ImportError:
    _HAS_LAZY = False
    LazyIndex = None
    write_lazy_index = None
    rewrite_lazy_index = None
    split_dyf3 = None
    from_faiss = None
    SearchResult = None
    AdaptiveProbeConfig = None
    ExtractedData = None
    StoredFieldValue = None
    StoredFieldInput = None
    TreeNode = None

# Fisher dimension weighting
import logging

# CatalogSpace — available via dyf.catalog (not re-exported)
# Pipeline DAG runner — available via dyf.pipeline (not re-exported)
# Concept graph — available via dyf.concept_graph (not re-exported)
from . import (
    catalog,  # noqa: F401
    concept_graph,  # noqa: F401
    pipeline,  # noqa: F401
)

# Tree-leaf agglomeration
from .agglomerate import agglomerate_tree_leaves, louvain_cluster_leaves, merge_to_max_k

# Categorical DAG
from .categorical import (
    AxisDiagnostic,
    CategoryGraph,
    coarsen,
    diagnose_axes,
    diagnostics_to_metadata,
    discover_categorical_columns,
    embed_with_diagnostics,
    load_category_graphs,
    multi_level_fisher_weights,
    store_category_graph,
)

# Cluster-tree DAG
from .cluster_tree import (
    build_cluster_tree_dag,
    compute_sibling_keywords,
    derive_path_labels,
    format_cluster_context,
)

# Spatial cluster coloring
from .colors import spatial_color_map, spatial_rgb_map, tree_rgb_map
from .fisher import (
    apply_fisher_weights,
    compute_fisher_weights,
)

# Ontology (DAG taxonomy extraction)
from .ontology import (
    DAGChain,
    DAGMiningResult,
    DAGTaxonomy,
    HubScoreResult,
    ROGLayer,
    ROGResult,
    UnifiedOntologyResult,
    build_dag_taxonomy,
    build_rog_ontology,
    build_unified_ontology,
    compute_hub_score,
    compute_neighbor_diversity,
    mine_dag_chains,
)

# Provenance tracking
from .provenance import (
    Provenance,
    check_compatible,
    create_provenance,
    file_hash,
    params_hash,
    provenance_from_dict,
    provenance_to_dict,
)

# RAG index
from .rag import (
    BridgeIndex,
    FacetDiverseResult,
    OrthogonalAnchorResult,
    SuperConnectorResult,
    diversify_by_facet,
    find_super_connectors,
    get_kmeans_init,
    select_orthogonal_anchors,
)

# Re-ranking
from .rerank import (
    rerank_bridge_boost,
    rerank_bridge_mmr,
    rerank_mmr,
    rerank_standard,
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

logging.getLogger(__name__).addHandler(logging.NullHandler())

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
    # DYF tree
    "build_dyf_tree",
    "refine_dyf_tree",
    "refine_clusters",
    # Unified tree-cutting dispatcher
    "cut_tree_to_labels",
    # Lazy index
    "LazyIndex",
    "write_lazy_index",
    "rewrite_lazy_index",
    "split_dyf3",
    "from_faiss",
    "SearchResult",
    "AdaptiveProbeConfig",
    "ExtractedData",
    "StoredFieldValue",
    "StoredFieldInput",
    "TreeNode",
    "detect_dyf_version",
    # Fisher dimension weighting
    "compute_fisher_weights",
    "apply_fisher_weights",
    # extract_fisher_labels — deprecated, use coarsen() directly
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
    # CatalogSpace — internal/experimental, import directly from dyf.catalog
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
    # LouvainHierarchy — internal type, import directly from dyf.agglomerate
    # Spatial cluster coloring
    "spatial_rgb_map",
    "spatial_color_map",
    "tree_rgb_map",
    # Pipeline DAG runner — internal/experimental, import directly from dyf.pipeline
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
    # Concept graph — dev tooling, import directly from dyf.concept_graph
]

def check_rust_available():
    """Check if Rust acceleration is available."""
    return _HAS_RUST
