"""
DYF - Density Yields Features

Discover structure in embedding spaces using PCA-based LSH, and index it for search.

Finding your way around
-----------------------
There are over a hundred public names — too many to scan. `dyf.overview()` prints them
grouped with an entry point per group, `dyf.overview("trees")` lists one group, and
`dyf.overview(as_dict=True)` returns the same thing parseably. From a shell: `dyf api`.

    >>> import dyf
    >>> print(dyf.overview())

(The exact count is deliberately not written here — it would go stale. `overview()`
computes it, and a test asserts the map covers `__all__` exactly.)

The groups, and the one name to read first in each:

    density      DensityClassifier          dense / bridge / orphan by bucket density
    trees        build_dyf_tree             the k-ary tree everything routes through
    index        LazyIndex                  write and read .dyf files
    search       DenseSearchIndex           in-memory search over a tree
    retrieval    BridgeIndex                bridge-anchored ANN, and re-ranking
    dedup        near_duplicate_clusters    collapse near-duplicates before indexing
    ontology     build_unified_ontology     discover DAG taxonomies from embeddings
    catalogs     CatalogSpace               match against declared hierarchies
    categorical  diagnose_axes              label hierarchies and axis diagnostics
    labeling     compute_split_keywords     human-readable names for splits/clusters
    grouping     agglomerate_tree_leaves    merge leaves into N groups
    chunks       chunk_redundancy           assess chunked-document corpora
    provenance   create_provenance          artifact identity
    color        spatial_rgb_map            structure to RGB for visualization

Not re-exported — import directly: `dyf.pipeline`, `dyf.concept_graph`.

Typical path
------------
    >>> from dyf import build_dyf_tree, write_lazy_index, LazyIndex
    >>> tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3)
    >>> write_lazy_index(tree, embeddings, "corpus.dyf", stored_fields={"title": titles})
    >>> with LazyIndex("corpus.dyf") as idx:
    ...     result = idx.search(query, k=10, nprobe=256)

From the shell, `dyf info corpus.dyf` describes an index without loading it.

Density metrics
---------------
The Rust core returns raw per-item metrics; classification is up to you:
- bucket_id: LSH bucket assignment
- bucket_size: Number of items in the bucket
- centroid_similarity: Cosine similarity to bucket centroid (0-1)
- isolation_score: How isolated the item is (top_k_sim - median_sim)

    >>> from dyf import DensityClassifier
    >>> classifier = DensityClassifier(embedding_dim=384)
    >>> classifier.fit(embeddings)
    >>> print(classifier.report())

    >>> from dyf import DensityClassifierFull, LabelerConfig
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


# Warn if the installed dyf-rs is below what this package requires. An editable install
# can silently keep a stale wheel — KNOWN_ISSUES #1 was exactly that, a venv running
# dyf-rs 0.5.0 against a newer constraint.
#
# The floor is READ FROM THE PYPROJECT REQUIREMENT, not hard-coded. It used to be a literal
# `(0, 7, 0)` and drifted: pyproject moved to >=0.10.0 while this stayed at 0.7.0, so the
# guard against stale installs sat *below* the requirement and could not fire — a venv
# running 0.7.0 against a >=0.10.0 constraint passed it silently (found 2026-09-05).
# A hard-coded copy of a number that lives somewhere else is a number that will drift.
def _dyf_rs_floor() -> tuple[int, ...] | None:
    """Minimum dyf-rs version, parsed from this package's own declared requirement.

    Note this reads the *installed distribution's* metadata, which is what makes it
    self-consistent — and also means a stale editable install reports a stale floor.
    That is the correct failure direction: the remedy in both cases is
    `uv pip install -e .`, which the warning names.
    """
    try:
        import re as _re
        from importlib.metadata import requires as _requires

        for _req in _requires("dyf") or []:
            # Requirement strings have no space before the specifier: "dyf-rs>=0.7.0".
            name_match = _re.match(r"^\s*([A-Za-z0-9._-]+)", _req)
            if not name_match:
                continue
            if name_match.group(1).strip().lower().replace("_", "-") != "dyf-rs":
                continue
            version_match = _re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", _req)
            if version_match:
                return tuple(int(x) for x in version_match.group(1).split(".")[:3])
    except Exception:  # noqa: BLE001 — a version check must never break the import
        pass
    return None


if _HAS_RUST:
    import warnings as _warnings

    _DYF_RS_FLOOR = _dyf_rs_floor()
    _rs_ver_str = getattr(__import__("dyf_rs"), "__version__", None)
    if _rs_ver_str and _DYF_RS_FLOOR:
        try:
            _rs_ver = tuple(int(x) for x in _rs_ver_str.split(".")[:3])
            if _rs_ver < _DYF_RS_FLOOR:
                _warnings.warn(
                    f"dyf-rs {_rs_ver_str} is below the required "
                    f"{'.'.join(str(x) for x in _DYF_RS_FLOOR)} (from dyf's own dependency "
                    f"declaration). Run `uv pip install -e .` to refresh editable-install "
                    f"metadata.",
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

# Tree cutting. Was a dispatcher over two tree shapes until pca_tree was dropped.
from .cut import cut_tree_to_labels

# Ingest-time near-duplicate detection
from .dedup import (
    DedupForIndexResult,
    DedupResult,
    decode_members,
    dedup_for_index,
    near_duplicate_clusters,
)

# Dense in-memory multiprobe search (Rust-backed; see pyproject for the dyf-rs floor)
from .dense_search import DenseSearchIndex, flatten_tree

# DYF tree (recursive k-ary LSH splits) and boundary persistence.
#
# The two boundary-persistence names resolved to the `pca_tree` variants until
# 2026-09-05, so the natural call — build with `build_dyf_tree`, analyse with the
# top-level `extract_boundary_persistence` — died with a bare `KeyError: 'left'`.
# `pca_tree` is gone; these work on the tree this package actually produces.
from .dyf_tree import (
    boundary_persistence_scores,
    build_dyf_tree,
    extract_boundary_persistence,
    refine_clusters,
    refine_dyf_tree,
)

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

# CatalogSpace — substrate API, promoted to top-level export
# Pipeline DAG runner — available via dyf.pipeline (not re-exported)
# Concept graph — available via dyf.concept_graph (not re-exported)
from . import (
    catalog,  # noqa: F401
    concept_graph,  # noqa: F401
    pipeline,  # noqa: F401
)

# Grouped index of the public API. `tests/test_api_map.py` asserts it covers __all__
# exactly, so it cannot drift out of sync with the list below.
from ._api_map import API_GROUPS, overview

# Tree-leaf agglomeration
from .agglomerate import (
    LeafGroupingResult,
    agglomerate_tree_leaves,
    louvain_cluster_leaves,
    merge_to_max_k,
)
from .catalog import (
    CatalogConfig,
    CatalogMatch,
    CatalogSpace,
    CrossMapping,
    FittedCatalog,
    JointMatchResult,
    compute_similarity_entropy,
)

# Categorical DAG
from .categorical import (
    AxisDiagnostic,
    AxisDiagnosticsResult,
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
    # API introspection — start here
    "overview",
    "API_GROUPS",
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
    # Ingest-time near-duplicate detection
    "DedupForIndexResult",
    "DedupResult",
    "decode_members",
    "dedup_for_index",
    "near_duplicate_clusters",
    # PCA tree
    "extract_boundary_persistence",
    "boundary_persistence_scores",
    # DYF tree
    "build_dyf_tree",
    "refine_dyf_tree",
    "refine_clusters",
    # Dense Rust-backed search
    "DenseSearchIndex",
    "flatten_tree",
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
    "AxisDiagnosticsResult",
    "CategoryGraph",
    "coarsen",
    "diagnose_axes",
    "diagnostics_to_metadata",
    "discover_categorical_columns",
    "embed_with_diagnostics",
    "multi_level_fisher_weights",
    "store_category_graph",
    "load_category_graphs",
    # CatalogSpace — substrate API for hierarchical catalog matching
    "CatalogSpace",
    "CatalogConfig",
    "CatalogMatch",
    "CrossMapping",
    "FittedCatalog",
    "JointMatchResult",
    "compute_similarity_entropy",
    # (CatalogSpace public methods: get_fitted, get_lca_depth,
    #  get_cross_domain_affinity)
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
    "LeafGroupingResult",
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
